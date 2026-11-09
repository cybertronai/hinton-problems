# Fixed full-width MultiMNIST rerun

Run these Bash commands in the same shell session, from `multimnist-capsnet/`
in this contribution's exact checkout, using a private Python environment and fresh `artifacts/`, `data/`
and `run/` paths. The recorded CPU controls used Python 3.13.14 and
PyTorch 2.14.0+cu130. Data preparation also needs Pillow and PyArrow; obtain
compatible public packages separately and retain the environment lock. These
instructions do not install packages, launch a watchdog or grant execution.

This is seed **20261107**, `author2017`, FP32/no AMP/TF32 disabled, batch 128,
one 60,000,000-pair exposure (**468,750 optimizer updates**), then a separate
10,000,000-pair final-checkpoint test. No sweep or diagnostic-best selection is
part of this condition. See [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) for its
differences from the paper equations and historical reduced NumPy result.

## Public inputs to an offline manifest

The [pinned MNIST provider card](https://huggingface.co/datasets/ylecun/mnist/blob/77f3279092a1c1579b2250db8eafed0ad422088c/README.md)
declares MIT. Attribute MNIST to Yann LeCun, Corinna Cortes and Christopher J. C.
Burges. The bridge below is the public, Unlicense-released
[MNIST materializer](https://github.com/Sinkleberg/schmidhuber-problems/blob/4218bb53534dd3ed9b05a3416cd29f2a0e426965/mnist-deep-mlp/paper_inputs.py),
with its [license](https://github.com/Sinkleberg/schmidhuber-problems/blob/4218bb53534dd3ed9b05a3416cd29f2a0e426965/LICENSE).
It compares **every** decoded provider image/label to original IDX bytes/order,
checks exact headers, lengths, labels, counts and pinned Parquet SHA256, then
writes the raw files and `MANIFEST.json`. This uses test bytes only for identity
checking, never inference, selection or a score. No data or checkpoints belong
in a code contribution.

```bash
set -euo pipefail
mkdir artifacts run
mkdir artifacts/tools
git rev-parse HEAD > run/SOURCE_COMMIT.txt
idx=https://ossci-datasets.s3.amazonaws.com/mnist
for file in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
            t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl --fail --location "$idx/$file" --output "artifacts/$file"
done
provider=https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist
curl --fail --location "$provider/train-00000-of-00001.parquet" --output artifacts/train-00000-of-00001.parquet
curl --fail --location "$provider/test-00000-of-00001.parquet" --output artifacts/test-00000-of-00001.parquet
bridge=https://raw.githubusercontent.com/Sinkleberg/schmidhuber-problems/4218bb53534dd3ed9b05a3416cd29f2a0e426965
curl --fail --location "$bridge/mnist-deep-mlp/paper_inputs.py" --output artifacts/tools/paper_inputs.py
curl --fail --location "$bridge/LICENSE" --output artifacts/tools/LICENSE
python -s -B - <<'PY'
from pathlib import Path
import hashlib
import sys
bridge = Path('artifacts/tools/paper_inputs.py')
if hashlib.sha256(bridge.read_bytes()).hexdigest() != '7ac20f63b42b6089b3fb8400f9f81e905e1bf8a44bbb22d1c18658e52abf4a98':
    raise SystemExit('Pinned materializer bytes differ')
sys.path.insert(0, str(bridge.parent.resolve()))
from paper_inputs import materialize
p = Path('artifacts')
materialize(
    {'train': p/'train-00000-of-00001.parquet',
     'test': p/'test-00000-of-00001.parquet'},
    {'train': [p/'train-images-idx3-ubyte.gz', p/'train-labels-idx1-ubyte.gz'],
     'test': [p/'t10k-images-idx3-ubyte.gz', p/'t10k-labels-idx1-ubyte.gz']},
    Path('data'))
PY
DATA_SHA=$(python -s -B -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('data/MANIFEST.json').read_bytes()).hexdigest())")
```

The bridge refuses an existing `data/` directory. `paper_data.py` additionally
requires the four canonical raw stream SHA256 values; mismatches stop the run.
Keep the materializer in `artifacts/tools`, outside the trainer's `paper_*.py`
source identity. Do not edit code, data, protocol or the environment during an
arm or resume.

## Environment identity and controls

Use the same environment throughout. Record its Python executable, versions
and dependency lock, then hash that record. `--runtime-sha256` binds the supplied
identity for continuation; the trainer does not independently authenticate
installed dependency bytes. Operators must verify their actual runtime.

```bash
python -s -B -m pip freeze --all > run/PACKAGES.txt
python -s -B - <<'PY'
from pathlib import Path
import hashlib
import json
import platform
import sys
record = {'python': sys.version, 'platform': platform.platform(),
          'executableSHA256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
          'packagesSHA256': hashlib.sha256(Path('run/PACKAGES.txt').read_bytes()).hexdigest()}
Path('run/RUNTIME.json').write_text(json.dumps(record, sort_keys=True) + '\n')
PY
RUNTIME_SHA=$(python -s -B -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('run/RUNTIME.json').read_bytes()).hexdigest())")
python -s -B -m unittest discover -s . -p 'test_paper_*.py' -v
python -s -B paper_protocol.py
PURELIB=$(python -s -B -c "import sysconfig; print(sysconfig.get_path('purelib'))")
PLATLIB=$(python -s -B -c "import sysconfig; print(sysconfig.get_path('platlib'))")
CUDA_VISIBLE_DEVICES=-1 python -I -B -S - "$PURELIB" "$PLATLIB" <<'PY'
from pathlib import Path
import runpy
import sys
for path in sys.argv[1:]:
    sys.path.append(str(Path(path).resolve()))
sys.argv = ['cpu_reference_gate.py', '--output', 'run/cpu-controls']
runpy.run_path('cpu_reference_gate.py', run_name='__main__')
PY
```

The CPU gate deliberately requires isolated/no-site/no-bytecode flags. Its
wrapper explicitly adds this environment's package directories, without running
site hooks. Ordinary trainer commands use `-s -B` so their local imports and
private environment packages remain visible. With bare `-I -S paper_train.py`,
the trainer restores its local module path but not environment site-packages. CPU controls use synthetic tensors, not MNIST inference. Their
`RESULT.json` is a numerical/control receipt, not an official model result.

## Fixed train, then final-only test

Run only with adequate externally supervised resources and one chosen GPU.
There is no automatic device fallback, timeout or thermal watchdog in this CLI.
Power and thermal limits are external operator settings, not model
hyperparameters or settings changed by these commands.

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -s -B paper_train.py --phase train --device cuda:0 \
  --data-root data --manifest-sha256 "$DATA_SHA" \
  --runtime-sha256 "$RUNTIME_SHA" --output run
# Only after training has completed its fixed budget:
python -s -B paper_train.py --phase test --device cuda:0 \
  --data-root data --manifest-sha256 "$DATA_SHA" \
  --runtime-sha256 "$RUNTIME_SHA" --output run
```

The test phase authenticates the complete final training checkpoint before
opening the test split. It evaluates final weights, never the diagnostic best.
Training status and online losses are not final-model or held-out scores.
`run/TEST_RESULT.json` exists only after completed evaluation; report its raw
pair/digit and reconstruction denominators separately. The test journal counts
evaluation batches, not training optimizer updates.

The public CLI supports graceful resume as described next. The measured
experiment used separately reviewed single-use continuation admissions for
the same fixed test after an executor exception and a gaming stop; public CLI
resumability does not authorize a fresh admission, seed, trial or reset. Actual training continued from a committed 256-update/32,768-pair
prefix. These are execution-provenance distinctions, not additional CLI flags.
See [source identity and execution boundaries](PAPER_PROTOCOL.md#source-identity-and-execution-boundaries)
for the executed versus public LF digests. An existing executed checkpoint
cannot be resumed under the public LF source binding.

An ordinary empty `run/STOP` requests a checkpointed stop. After a successful
graceful stop, resolve that flag and repeat the same phase command with the
unchanged source/data/runtime and output; it resumes the authenticated cursor.
An unexpected exception/kill preserves its session lock: inspect and reconcile
it, never delete a lock or reset the test journal to force a rerun. Do not
overwrite `TEST_RESULT.json`. Retain checkpoints and raw data locally.

Completed CPU controls are documented in [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md).
See [the run record](PAPER_RUN.md) for the completed training exposure,
checkpoint identity and held-out evaluation status.
