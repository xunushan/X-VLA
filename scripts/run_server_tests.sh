#!/usr/bin/env bash
# 服务器 X-VLA 测试编排：环境 -> 模型 -> 数据 -> 模块测试 -> 训练冒烟 -> resume。
#
# 覆盖（docs/服务器测试计划.md 的落地执行）：
#   0. 环境检查（GPU/conda/磁盘/torch cuda）
#   1. 模型下载 + 校验（config action_mode=ee6d/num_actions=30、safetensors 大小）
#   2. 数据 16d->6d 转换 + check_data.py 验证 + splits 训练索引过滤（meta.json episodes）
#   3. pytest 模块测试（handler / dataset_reader / make_goai_20d）
#   4. 训练冒烟 120 步：batch=4 × accum=8 → effective_batch=32；前 30 步冻结 vlm+core 后全参数；
#      日志格式、DATA_PCT、视频解码计时聚合、模型文件大小
#   5. resume：global_step / optimizer 恢复、loss 连续性、state.json 复核
#   6. resume x 冻结阶段交错：冻结期 ckpt 内 resume，lr_vlm 0 -> 非 0
#
# 用法：
#   bash scripts/run_server_tests.sh                  # 全跑
#   bash scripts/run_server_tests.sh --only 5         # 只跑 resume 阶段
#   SKIP=3 bash scripts/run_server_tests.sh           # 跳过 pytest
#
# 磁盘注意：每 ckpt ≈ 权重 3.5G + AdamW 状态 7G ≈ 11G；阶段 5 峰值 2 ckpt ≈ 22G。
#   服务器 / 盘只剩 ~22G 时，先清理 /data 或用 XVLA_OUT_BASE 指到更大磁盘。
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PROJECT_ROOT   # 阶段子 shell 需要

# ---- 可覆盖配置（export 供子 shell 阶段使用）----
export CONDA_ENV="${XVLA_CONDA_ENV:-xvla}"
export DATA_ROOT="${XVLA_DATA_ROOT:-/data/data/lerobot_v30_ee_6d}"      # 转换后 6d 数据（splits 引用此路径）
export SRC_16="${XVLA_SRC_16:-/data/data/lerobot_v30_ee}"               # 服务器现有 16d 数据
export SPLIT_FILE="${XVLA_SPLIT_FILE:-/data/splits/lerobot_v30_ee_6d_train90_seed42.json}"
export META_JSON="${XVLA_META_JSON:-${DATA_ROOT}/meta.json}"
export MODEL_DIR="${XVLA_MODEL_DIR:-/data/checkpoints/xvla/X-VLA-Pt}"
export OUT_BASE="${XVLA_OUT_BASE:-${PROJECT_ROOT}/runnings/server_tests}"

# ---- 训练参数（与用户对齐：batch=4 × accum=8 → effective_batch=32；冒烟 120 步/freeze 30）----
export TRAIN_BATCH_SIZE="${XVLA_TRAIN_BATCH_SIZE:-4}"
export TRAIN_ACCUM="${XVLA_TRAIN_ACCUM:-8}"
export TRAIN_WORKERS="${XVLA_TRAIN_WORKERS:-4}"

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
  echo "=== disk (/data) ==="
  df -h /data | tail -1
  FREE_GB=$(df -BG --output=avail /data | tail -1 | tr -d 'G')
  echo "  [INFO] /data free: ${FREE_GB}G (每 ckpt 约 11G；阶段 5 峰值需 ~22G)"
  if [ "${FREE_GB}" -lt 26 ] 2>/dev/null; then
    echo "  [WARN] 剩余磁盘 < 26G，阶段 5/6 可能因磁盘不足失败。请先清理 /data 或用 XVLA_OUT_BASE 指到更大磁盘。"
  fi
  echo "=== 关键数据/模型存在性 ==="
  [ -d "${DATA_ROOT}" ] && echo "  6d data: ${DATA_ROOT} (已存在)" || echo "  6d data: ${DATA_ROOT} (不存在，阶段 2 会转换)"
  [ -d "${SRC_16}" ] && echo "  16d data: ${SRC_16} ✓" || echo "  16d data: ${SRC_16} ✗"
  [ -f "${SPLIT_FILE}" ] && echo "  splits: ${SPLIT_FILE} ✓" || echo "  splits: ${SPLIT_FILE} ✗"
  [ -f "${MODEL_DIR}/model.safetensors" ] && echo "  model: ${MODEL_DIR} ✓" || echo "  model: ${MODEL_DIR} ✗"
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
print("  [INFO] preprocessor 只处理文本（encode_language）；图像预处理走 dataset.image_aug。")
print("        此处 ImageNet 统计仅用于后续推理代码对齐，不影响训练（见测试计划 2.5）。")
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
    bash "${PROJECT_ROOT}/scripts/prepare_data.sh" "${SRC_16}" "${DATA_ROOT}" || { echo "[FAIL] prepare_data failed"; exit 1; }
  else
    echo "== ${DATA_ROOT} exists, skip conversion (需要强制重建：先删 ${DATA_ROOT}) =="
  fi
  PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/scripts/check_data.py" "${DATA_ROOT}" "${META_JSON}"
  echo "--- 训练集索引过滤（splits） ---"
  python - "${META_JSON}" "${SPLIT_FILE}" <<'PY'
import json, sys
meta, split_file = sys.argv[1:3]
m = json.load(open(meta))
eps = m.get("episodes")
if eps is None:
    print("  [FAIL] meta.json episodes=None（训练未按训练集过滤）；检查 prepare_data.sh 的 splits 读取")
    raise SystemExit(1)
print(f"  meta.json episodes: {len(eps)} 个（应为 1080，train90_seed42）")
if len(eps) != 1080:
    print("  [WARN] episodes 数量 != 1080，与 train90 划分不符，确认 splits 文件/字段")
import sys as _s
_s = json.load(open(split_file))["train"]
assert eps == sorted(_s), "meta.json episodes != splits train 列表"
print("  [PASS] meta.json episodes 与 splits train 列表一致")
PY
  echo "== 2-data done =="
PHASE2
fi

# ================================================================ PHASE 3
if ! skip_phase 3; then
run_phase "3-pytest" <<'PHASE3'
  set -uo pipefail
  if [ ! -d "${SRC_16}" ]; then
    echo "  [SKIP] 16d 源数据 ${SRC_16} 不存在，跳过 pytest（见测试计划 2.5 说明）"
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

# ================================================================ PHASE 4（训练冒烟 120 步 + 日志/计时/模型大小）
if ! skip_phase 4; then
run_phase "4-train-smoke" <<'PHASE4'
  set -uo pipefail
  SMOKE_OUT="${OUT_BASE}/smoke"
  rm -rf "${SMOKE_OUT}"
  TRAIN_OUTPUT_DIR="${SMOKE_OUT}" TRAIN_ITERS=120 TRAIN_SAVE_INTERVAL=120 \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" TRAIN_ACCUM="${TRAIN_ACCUM}" \
    TRAIN_NUM_WORKERS="${TRAIN_WORKERS}" TRAIN_LOG_INTERVAL=10 \
    TRAIN_FREEZE_STEPS=30 TRAIN_WARMUP_STEPS=0 \
    TRAIN_TIMING_DIR="${SMOKE_OUT}/timing" \
    bash "${PROJECT_ROOT}/scripts/train.sh" > "${SMOKE_OUT}.cmd.log" 2>&1 \
    || { echo "  [FAIL] 冒烟训练 train.sh 退出非 0（见 ${SMOKE_OUT}.cmd.log，若是 CUDA OOM 说明 batch=${TRAIN_BATCH_SIZE} 超显存，可降 batch 或升 accum）"; exit 1; }
  LOG="${SMOKE_OUT}/train.log"
  [ -f "${LOG}" ] || { echo "  [FAIL] no train.log"; exit 1; }
  echo "--- train.log (tail -25) ---"
  tail -25 "${LOG}"
  echo "--- 日志格式检查 ---"
  check_log_line "${LOG}" "^[0-9]{2}:[0-9]{2}:[0-9]{2} \| [A-Z]+ \| train \| " "日志行格式 asctime|level|name|message"
  check_log_line "${LOG}" "\[[0-9]+/120\] loss=[0-9.]+ lr_core=[0-9.e-]+ lr_vlm=[0-9.e-]+" "step 日志含 loss/lr_core/lr_vlm（/120）"
  check_log_line "${LOG}" "effective_batch=32" "effective_batch=4×1×8=32（batch×world×accum）"
  check_log_line "${LOG}" "DATA_PCT=[0-9]+%" "每 step 数据预处理时间占比 DATA_PCT"
  check_log_line "${LOG}" "DECODE timing:.*ms/sample" "视频解码计时聚合（decode ms/样本 + fps + %）"
  check_log_line "${LOG}" "USED_CPU=[0-9.e+-]+ .*USED_GPU=[0-9.e+-]+" "资源监控 USED_CPU/USED_GPU"
  echo "--- 冻结/解冻 lr 校验（前 30 步 lr_vlm=0，之后 >0） ---"
  before=$(grep -E "\[(10|20|30)/120\]" "${LOG}" | grep -oE "lr_vlm=[0-9.e+-]+" | head -1)
  after=$(grep -E "\[(40|60|90|120)/120\]" "${LOG}" | grep -oE "lr_vlm=[0-9.e+-]+" | head -1)
  echo "  before(step<30) lr_vlm=${before}  after(step>=30) lr_vlm=${after}"
  if echo "${before}" | grep -qE "lr_vlm=0.00e\+00$" && echo "${after}" | grep -qE "lr_vlm=[^0]\.[0-9]+e"; then
    echo "  [PASS] 冻结边界正确：step<30 lr_vlm=0，step>=30 全参数 lr_vlm>0（lr 恒定）"
    log "  CHECK PASS: 冻结边界 lr_vlm 0->非0（freeze=30 全参数训练）"
  else
    echo "  [FAIL] 冻结/解冻 lr 异常 (before='${before}' after='${after}')"
    touch "${CHECK_MARKER}"
  fi
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
  rm -rf "${OUT_BASE}/smoke"   # 释放阶段 4 的 ckpt（~11G），本阶段峰值 2 ckpt ≈ 22G
  RES_OUT="${OUT_BASE}/resume"
  rm -rf "${RES_OUT}"
  TRAIN_OUTPUT_DIR="${RES_OUT}" TRAIN_ITERS=60 TRAIN_SAVE_INTERVAL=60 \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" TRAIN_ACCUM="${TRAIN_ACCUM}" \
    TRAIN_NUM_WORKERS="${TRAIN_WORKERS}" TRAIN_LOG_INTERVAL=10 \
    TRAIN_FREEZE_STEPS=30 TRAIN_WARMUP_STEPS=0 \
    bash "${PROJECT_ROOT}/scripts/train.sh" > "${RES_OUT}.run1.log" 2>&1 \
    || { echo "  [FAIL] resume run1 训练退出非 0（见 ${RES_OUT}.run1.log）"; exit 1; }
  TRAIN_OUTPUT_DIR="${RES_OUT}" TRAIN_ITERS=120 TRAIN_SAVE_INTERVAL=120 \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" TRAIN_ACCUM="${TRAIN_ACCUM}" \
    TRAIN_NUM_WORKERS="${TRAIN_WORKERS}" TRAIN_LOG_INTERVAL=10 \
    TRAIN_FREEZE_STEPS=30 TRAIN_WARMUP_STEPS=0 \
    bash "${PROJECT_ROOT}/scripts/train.sh" --resume latest > "${RES_OUT}.run2.log" 2>&1 \
    || { echo "  [FAIL] resume run2 训练退出非 0（见 ${RES_OUT}.run2.log）"; exit 1; }
  LOG="${RES_OUT}/train.log"
  echo "--- resume 关键日志 ---"
  grep -E "Resume|continue from global_step|Restored" "${LOG}" | tail -5
  check_log_line "${LOG}" "Resume: continue from global_step=60" "resume 从 global_step=60 续跑"
  LAST_CK=$(ls -d "${RES_OUT}"/ckpt-* | sort -V | tail -1)
  gs=$(python -c "import json;print(json.load(open('$LAST_CK/state.json'))['global_step'])")
  if [ -f "${LAST_CK}/optimizer.pt" ] && [ "${gs}" = "120" ]; then
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
  rm -rf "${OUT_BASE}/smoke" "${OUT_BASE}/resume"   # 释放前序阶段 ckpt，峰值 2 ckpt ≈ 22G
  FR_OUT="${OUT_BASE}/freeze_resume"
  rm -rf "${FR_OUT}"
  # run1：20 步全冻结期保存 ckpt-20（vlm/core 无梯度 → optimizer 状态小，省磁盘）；
  # run2：从 ckpt-20 resume，跨过 step30 冻结边界续到 120，验证解冻后 lr_vlm 0->非0
  TRAIN_OUTPUT_DIR="${FR_OUT}" TRAIN_ITERS=20 TRAIN_SAVE_INTERVAL=20 \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" TRAIN_ACCUM="${TRAIN_ACCUM}" \
    TRAIN_NUM_WORKERS="${TRAIN_WORKERS}" TRAIN_LOG_INTERVAL=10 \
    TRAIN_FREEZE_STEPS=30 TRAIN_WARMUP_STEPS=0 \
    bash "${PROJECT_ROOT}/scripts/train.sh" > "${FR_OUT}.run1.log" 2>&1 \
    || { echo "  [FAIL] freeze run1 训练退出非 0（见 ${FR_OUT}.run1.log）"; exit 1; }
  TRAIN_OUTPUT_DIR="${FR_OUT}" TRAIN_ITERS=120 TRAIN_SAVE_INTERVAL=120 \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" TRAIN_ACCUM="${TRAIN_ACCUM}" \
    TRAIN_NUM_WORKERS="${TRAIN_WORKERS}" TRAIN_LOG_INTERVAL=10 \
    TRAIN_FREEZE_STEPS=30 TRAIN_WARMUP_STEPS=0 \
    bash "${PROJECT_ROOT}/scripts/train.sh" --resume latest > "${FR_OUT}.run2.log" 2>&1 \
    || { echo "  [FAIL] freeze run2 训练退出非 0（见 ${FR_OUT}.run2.log）"; exit 1; }
  LOG="${FR_OUT}/train.log"
  echo "--- resume 后 lr_vlm 变化（应 0 -> 非 0）---"
  grep -oE "\[[0-9]+/120\] .*lr_vlm=[0-9.e-]+" "${LOG}" | tail -12
  # run1 段（iters=20）全冻结：lr_vlm=0；run2 段 step>=30 解冻：lr_vlm>0
  before=$(grep -E "\[(10|20)/20\]" "${LOG}" | grep -oE "lr_vlm=[0-9.e+-]+" | head -1)
  after=$(grep -E "\[(40|60|90|120)/120\]" "${LOG}" | grep -oE "lr_vlm=[0-9.e+-]+" | head -1)
  echo "  before(step<30) lr_vlm=${before}  after(step>=30) lr_vlm=${after}"
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
