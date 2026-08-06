#!/usr/bin/env bash
# 服务器 X-VLA 测试编排：环境 -> 模型 -> 数据 -> 模块测试 -> 训练冒烟 -> resume。
#
# 覆盖（docs/服务器测试计划.md 的落地执行）：
#   0. 环境检查（GPU/conda/磁盘/torch cuda）
#   1. 模型下载 + 校验（config action_mode=ee6d/num_actions=30、safetensors 大小）
#   2. 数据 16d->6d 转换 + check_data.py 验证（主表/stats/info/episodes/视频/handler 样本）
#   3. pytest 模块测试（handler / dataset_reader / make_goai_20d）
#   4. 训练冒烟：日志格式、模型文件大小、tensorboard 可写
#   5. resume：global_step / optimizer 恢复、loss 连续性、state.json 复核
#   6. resume x 冻结阶段交错：freeze_steps 前后各 resume 一次，lr_vlm 0 -> 非 0
#
# 用法：
#   bash scripts/run_server_tests.sh                  # 全跑
#   bash scripts/run_server_tests.sh --only 5         # 只跑 resume 阶段
#   SKIP=3 bash scripts/run_server_tests.sh           # 跳过 pytest
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PROJECT_ROOT   # 阶段子 shell 需要

# ---- 可覆盖配置（export 供子 shell 阶段使用）----
export CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
export DATA_ROOT="${XVLA_DATA_ROOT:-/data/lerobot_v30_ee_6d}"
export META_JSON="${XVLA_META_JSON:-${DATA_ROOT}/meta.json}"
export MODEL_DIR="${XVLA_MODEL_DIR:-/data/checkpoints/xvla/X-VLA-Pt}"
export OUT_BASE="${XVLA_OUT_BASE:-${PROJECT_ROOT}/runnings/server_tests}"

SUMMARY_LOG="${OUT_BASE}/summary.log"
mkdir -p "${OUT_BASE}"
: > "${SUMMARY_LOG}"
export SUMMARY_LOG

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------- helpers（export 给子 shell）
log() { echo "$*" >> "${SUMMARY_LOG}"; }
export -f log

# check_log_line：$1=log 文件 $2=regex $3=desc；命中写 PASS，否则写 FAIL 并 touch 失败标记
check_log_line() {
  if grep -qE "$2" "$1" 2>/dev/null; then
    log "  CHECK PASS: $3"
    echo "  [PASS] $3"
    return 0
  else
    log "  CHECK FAIL: $3  (regex '$2' not in '$1')"
    echo "  [FAIL] $3"
    touch "${CHECK_MARKER}"
    return 1
  fi
}
export -f check_log_line

skip_phase() {
  local p="$1"
  if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${p}" ]; then return 0; fi
  case ",${SKIP:-}," in *",${p},"*) return 0 ;; esac
  return 1
}

# run_phase：$1=阶段名；stdin 为阶段脚本（子 shell 运行，退出码即阶段成败）
run_phase() {
  local name="$1"
  export CHECK_MARKER="${OUT_BASE}/.check_fail_${name}"
  rm -f "${CHECK_MARKER}"
  echo "================================================================"
  echo "[run_tests] >>> PHASE ${name} (START)"
  echo "================================================================"
  local rc=0
  bash -s || rc=$?
  if [ "${rc}" -eq 0 ] && [ ! -f "${CHECK_MARKER}" ]; then
    log "  PHASE ${name}: OK"
    echo "[run_tests] <<< PHASE ${name} (OK)"
  else
    PHASE_FAILS+=("${name}")
    log "  PHASE ${name}: FAIL"
    echo "[run_tests] <<< PHASE ${name} (FAIL)"
  fi
}

# ================================================================ PHASE 0
if ! skip_phase 0; then
run_phase "0-env" <<'PHASE0'
  set -uo pipefail
  echo "=== GPU ==="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo "=== conda ==="
  conda env list
  echo "=== torch ==="
  python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available(), '| dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  echo "=== disk ==="
  df -h /data | tail -1
  echo "== 0-env done =="
PHASE0
fi

# ================================================================ PHASE 1
if ! skip_phase 1; then
run_phase "1-model" <<'PHASE1'
  set -uo pipefail
  if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "== model not found, downloading =="
    bash "${PROJECT_ROOT}/scripts/download_model.sh" || { echo "[FAIL] download failed"; exit 1; }
  fi
  ls -lh "${MODEL_DIR}/model.safetensors"
  SIZE_MB=$(du -m "${MODEL_DIR}/model.safetensors" | cut -f1)
  echo "  [${SIZE_MB} MB] model.safetensors"
  if [ "${SIZE_MB}" -lt 3000 ]; then
    echo "  [FAIL] model.safetensors too small: ${SIZE_MB}MB < 3000MB"
    exit 1
  fi
  log "  CHECK PASS: model.safetensors size ${SIZE_MB} MB"
  python - "${MODEL_DIR}" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/config.json"))
print(f"  config: action_mode={d.get('action_mode')} num_actions={d.get('num_actions')}")
assert d.get("action_mode") == "ee6d", "expect pretrained action_mode=ee6d"
assert d.get("num_actions") == 30, "expect num_actions=30"
PY
  python - "${MODEL_DIR}" <<'PY'
import json, sys
p = json.load(open(f"{sys.argv[1]}/preprocessor_config.json"))
print(f"  preprocessor image_mean={p.get('image_mean')} image_std={p.get('image_std')} type={p.get('image_processor_type')}")
assert p.get("image_mean") == [0.485, 0.456, 0.406], "expect ImageNet mean (aligned with training)"
PY
  echo "== 1-model done =="
PHASE1
fi

# ================================================================ PHASE 2
if ! skip_phase 2; then
run_phase "2-data" <<'PHASE2'
  set -uo pipefail
  if [ ! -d "${DATA_ROOT}" ]; then
    echo "== ${DATA_ROOT} missing, converting from 16d =="
    bash "${PROJECT_ROOT}/scripts/prepare_data.sh" || { echo "[FAIL] prepare_data failed"; exit 1; }
  else
    echo "== ${DATA_ROOT} exists, skip conversion (需要强制重建：先删 ${DATA_ROOT}) =="
  fi
  PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/scripts/check_data.py" "${DATA_ROOT}" "${META_JSON}"
  echo "== 2-data done =="
PHASE2
fi

# ================================================================ PHASE 3
if ! skip_phase 3; then
run_phase "3-pytest" <<'PHASE3'
  set -uo pipefail
  SRC16="${DATA_ROOT/6d/ee}"
  if [ ! -d "${SRC16}" ]; then
    echo "  [SKIP] 16d 源数据 ${SRC16} 不存在，跳过 pytest（见测试计划 2.5 说明）"
    echo "  [INFO] pytest skipped"
    exit 0
  fi
  PYTEST_OUT="${OUT_BASE}/pytest.out"
  XVLA_DATA_ROOT="${DATA_ROOT}" python -m pytest \
    test/test_handler.py test/test_dataset_reader.py test/test_make_goai_20d.py -q \
    > "${PYTEST_OUT}" 2>&1 || { tail -30 "${PYTEST_OUT}"; echo "  [FAIL] pytest 退出非 0"; exit 1; }
  tail -15 "${PYTEST_OUT}"
  echo "== 3-pytest done =="
PHASE3
fi

# ================================================================ PHASE 4（训练冒烟 + 日志格式 + 模型大小）
if ! skip_phase 4; then
run_phase "4-train-smoke" <<'PHASE4'
  set -uo pipefail
  SMOKE_OUT="${OUT_BASE}/smoke"
  rm -rf "${SMOKE_OUT}"
  TRAIN_OUTPUT_DIR="${SMOKE_OUT}" TRAIN_ITERS=30 TRAIN_SAVE_INTERVAL=15 \
    TRAIN_BATCH_SIZE=1 TRAIN_NUM_WORKERS=4 TRAIN_LOG_INTERVAL=5 \
    bash "${PROJECT_ROOT}/scripts/train.sh" > "${SMOKE_OUT}.cmd.log" 2>&1 \
    || { echo "  [FAIL] 冒烟训练 train.sh 退出非 0（见 ${SMOKE_OUT}.cmd.log）"; exit 1; }
  LOG="${SMOKE_OUT}/train.log"
  [ -f "${LOG}" ] || { echo "  [FAIL] no train.log"; exit 1; }
  echo "--- train.log (tail -25) ---"
  tail -25 "${LOG}"
  echo "--- 日志格式检查 ---"
  check_log_line "${LOG}" "^[0-9]{2}:[0-9]{2}:[0-9]{2} \| [A-Z]+ \| train \| " "日志行格式 asctime|level|name|message"
  check_log_line "${LOG}" "\[[0-9]+/30\] loss=[0-9.]+ lr_core=[0-9.e-]+ lr_vlm=[0-9.e-]+" "step 日志含 loss/lr_core/lr_vlm"
  check_log_line "${LOG}" "effective_batch=[0-9]+" "打印 effective_batch（=batch*world*accum）"
  check_log_line "${LOG}" "USED_CPU=[0-9.e-]+ .*USED_GPU=[0-9.e-]+" "资源监控 USED_CPU/USED_GPU"
  echo "--- 模型文件大小 ---"
  for ck in "${SMOKE_OUT}"/ckpt-*; do
    [ -d "$ck" ] || continue
    echo "  $(basename "$ck"): $(du -sh "$ck" | cut -f1) total, $(ls -lh "$ck/model.safetensors" | awk '{print $5}') safetensors"
  done
  pretrain_mb=$(du -m "${MODEL_DIR}/model.safetensors" | cut -f1)
  last_ck=$(ls -d "${SMOKE_OUT}"/ckpt-* | sort -V | tail -1)
  ck_mb=$(du -m "${last_ck}/model.safetensors" | cut -f1)
  ratio=$(python -c "print(f'{$ck_mb/$pretrain_mb:.2f}')")
  echo "  [INFO] ckpt/preTrain size ratio=${ratio} (ck=${ck_mb}MB pre=${pretrain_mb}MB)"
  python - <<PY
ok = 0.8 <= ${ratio} <= 1.2
print(f"  [{'PASS' if ok else 'FAIL'}] 模型文件大小与预训练同量级 ratio=${ratio}")
if not ok: raise SystemExit(1)
PY
  echo "== 4-train-smoke done =="
PHASE4
fi

# ================================================================ PHASE 5（resume）
if ! skip_phase 5; then
run_phase "5-resume" <<'PHASE5'
  set -uo pipefail
  RES_OUT="${OUT_BASE}/resume"
  rm -rf "${RES_OUT}"
  TRAIN_OUTPUT_DIR="${RES_OUT}" TRAIN_ITERS=30 TRAIN_SAVE_INTERVAL=15 \
    TRAIN_BATCH_SIZE=1 TRAIN_NUM_WORKERS=4 TRAIN_LOG_INTERVAL=5 \
    bash "${PROJECT_ROOT}/scripts/train.sh" > "${RES_OUT}.run1.log" 2>&1 \
    || { echo "  [FAIL] resume run1 训练退出非 0（见 ${RES_OUT}.run1.log）"; exit 1; }
  TRAIN_OUTPUT_DIR="${RES_OUT}" TRAIN_ITERS=45 TRAIN_SAVE_INTERVAL=15 \
    TRAIN_BATCH_SIZE=1 TRAIN_NUM_WORKERS=4 TRAIN_LOG_INTERVAL=5 \
    bash "${PROJECT_ROOT}/scripts/train.sh" --resume latest > "${RES_OUT}.run2.log" 2>&1 \
    || { echo "  [FAIL] resume run2 训练退出非 0（见 ${RES_OUT}.run2.log）"; exit 1; }
  LOG="${RES_OUT}/train.log"
  echo "--- resume 关键日志 ---"
  grep -E "Resume|continue from global_step|Restored" "${LOG}" | tail -5
  check_log_line "${LOG}" "Resume: continue from global_step=30" "resume 从 global_step=30 续跑"
  LAST_CK=$(ls -d "${RES_OUT}"/ckpt-* | sort -V | tail -1)
  gs=$(python -c "import json;print(json.load(open('$LAST_CK/state.json'))['global_step'])")
  if [ -f "${LAST_CK}/optimizer.pt" ] && [ "${gs}" = "45" ]; then
    echo "  [PASS] final ckpt ${LAST_CK} global_step=${gs}, optimizer.pt 存在"
    log "  CHECK PASS: final ckpt global_step=${gs} + optimizer.pt"
  else
    echo "  [FAIL] final ckpt global_step=${gs} / optimizer.pt 缺失"
    touch "${CHECK_MARKER}"
  fi
  L1=$(grep -oE "loss=[0-9.]+" "${RES_OUT}.run1.log" | tail -1 | cut -d= -f2)
  L2=$(grep -oE "loss=[0-9.]+" "${RES_OUT}.run2.log" | head -1 | cut -d= -f2)
  echo "  [INFO] run1 last loss=${L1}  run2 first loss=${L2}"
  python - <<PY
try:
    r = float("${L2}") / max(float("${L1}"), 1e-9)
    ok = 0.1 < r < 10
except Exception:
    r, ok = 0.0, False
print(f"  [{'PASS' if ok else 'FAIL'}] resume loss 连续性 ratio={r:.3f} (0.1~10)")
if not ok: raise SystemExit(1)
PY
  echo "== 5-resume done =="
PHASE5
fi

# ================================================================ PHASE 6（resume x 冻结阶段交错）
if ! skip_phase 6; then
run_phase "6-freeze-resume" <<'PHASE6'
  set -uo pipefail
  FR_OUT="${OUT_BASE}/freeze_resume"
  rm -rf "${FR_OUT}"
  TRAIN_OUTPUT_DIR="${FR_OUT}" TRAIN_ITERS=20 TRAIN_SAVE_INTERVAL=10 \
    TRAIN_FREEZE_STEPS=20 TRAIN_BATCH_SIZE=1 TRAIN_NUM_WORKERS=4 TRAIN_LOG_INTERVAL=5 \
    bash "${PROJECT_ROOT}/scripts/train.sh" > "${FR_OUT}.run1.log" 2>&1 \
    || { echo "  [FAIL] freeze run1 训练退出非 0（见 ${FR_OUT}.run1.log）"; exit 1; }
  TRAIN_OUTPUT_DIR="${FR_OUT}" TRAIN_ITERS=35 TRAIN_SAVE_INTERVAL=10 \
    TRAIN_FREEZE_STEPS=20 TRAIN_BATCH_SIZE=1 TRAIN_NUM_WORKERS=4 TRAIN_LOG_INTERVAL=5 \
    bash "${PROJECT_ROOT}/scripts/train.sh" --resume latest > "${FR_OUT}.run2.log" 2>&1 \
    || { echo "  [FAIL] freeze run2 训练退出非 0（见 ${FR_OUT}.run2.log）"; exit 1; }
  LOG="${FR_OUT}/train.log"
  echo "--- resume 后 lr_vlm 变化（应 0 -> 非 0）---"
  grep -oE "\[[0-9]+/35\] .*lr_vlm=[0-9.e-]+" "${LOG}" | tail -12
  # run1 段（iters=20）step<20 冻结：lr_vlm=0；run2 段（iters=35）step>=20 解冻：lr_vlm>0
  before=$(grep -E "\[(10|15)/20\]" "${LOG}" | grep -oE "lr_vlm=[0-9.e-]+" | head -1)
  after=$(grep -E "\[(25|30|35)/35\]" "${LOG}" | grep -oE "lr_vlm=[0-9.e-]+" | head -1)
  echo "  before(step<20) lr_vlm=${before}  after(step>=20) lr_vlm=${after}"
  if echo "${before}" | grep -qE "lr_vlm=0.00e\+00$" && echo "${after}" | grep -qE "lr_vlm=[^0]\.[0-9]+e"; then
    echo "  [PASS] resume 跨冻结边界：阶段一 lr_vlm=0，阶段二解冻 lr_vlm>0"
    log "  CHECK PASS: resume 跨冻结边界 lr_vlm 0->非0"
  else
    echo "  [FAIL] lr_vlm 冻结/解冻切换异常 (before='${before}' after='${after}')"
    touch "${CHECK_MARKER}"
  fi
  echo "== 6-freeze-resume done =="
PHASE6
fi

# ================================================================ summary
echo "================================================================"
echo "[run_tests] ===== SUMMARY ====="
cat "${SUMMARY_LOG}"
echo "================================================================"
if [ "${#PHASE_FAILS[@]}" -gt 0 ]; then
  echo "[run_tests] FAILED phases: ${PHASE_FAILS[*]}"
  echo "[run_tests] 日志目录: ${OUT_BASE}/"
  exit 1
fi
echo "[run_tests] ALL PHASES OK"
echo "[run_tests] 日志目录: ${OUT_BASE}/"
