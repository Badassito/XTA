# Native TTA detector scheduler

Each scalar TTA run requires one C++ scheduler for prepared detector admission,
dispatch, validated completions, byte credits, publication tickets, and GPU
stage admission. Python plans the scientific work, prepares the detector
inputs, runs model SDKs in isolated processes, and consumes detector results
for assembly, projection, SAM work, and output. A confidence sweep uses each
scalar child's scheduler for its detector work.

## Supported detector modes

| Configuration | Behavior |
| --- | --- |
| OpenVINO CPU only, for example `--device cpu` | Full-frame detector work uses the configured CPU actors. |
| CUDA only, for example `--device 0` or `--device 0,1` | Full-frame detector work uses the selected CUDA actors. Each ordinary D1 parent has one resident detector worker. |
| OpenVINO and CUDA together, for example `--device 0:cpu` | Preflight rejects hybrid detector dispatch. |
| OpenVINO CPU only with requested Azimuthal or Tilted Azimuthal views | Preflight rejects those detector views. |
| Dense detector tiles selected by `--enable_tile` | Preflight rejects the tile ownership route. |
| Multiple detector GPUs with `YOLO_TTA_D1_OWNER_GROUPS=1` | Preflight rejects multi-worker D1 owner groups. |

The full-frame inventory includes direct unions, file-backed results, and
ordinary single-owner D1. The adapter declares complete task ranges and
explicit policy siblings before admission. It checks that every range has an
eligible actor. A CPU detector run can still use explicitly selected SAM
devices for SAM processing. The restrictions above apply to detector
dispatch.

## Startup and installation

Use an XTA wheel built for your CPython ABI and platform, or launch from a
complete source checkout. The wheel includes `XTA._prepared_scheduler`. A
source launch builds or reuses the required extension in a cache outside the
checkout. Source builds require setuptools, matching CPython development
headers, and a C++17 toolchain. On Windows this means MSVC Build Tools and a
Windows SDK. On Linux the build uses a C++ compiler and pthread support.

Prebuild with the same interpreter that will run TTA. The command reports the
extension path and authenticates a cache hit against the current build inputs
and interpreter ABI.

```powershell
python -X utf8 -B tools/build_native_scheduler.py --cache-dir ../Scratch/Temp/xta-native-scheduler
```

Set `XTA_NATIVE_SCHEDULER_CACHE` to use that cache at launch. To select one
prebuilt source artifact, set `XTA_NATIVE_SCHEDULER_BINARY` to the printed
extension path and retain its `native_scheduler_build.json` receipt. TTA
refuses an invalid artifact or incompatible ABI. Use a valid wheel or rebuild
with the runtime interpreter and current checkout. Use a fresh external cache
directory if the previous build receipt or binary fails validation. See the
[packaging checks](maintenance.md#packaging-checks).

## Tickets, receipts, and retirement

1. Python declares the known detector inventory and byte requirements. For an
   immutable descriptor-only parent, Python registers every task range before
   funding. Native admission moves that full inventory directly to its ready
   queue when the credit limits permit. Parents that need backing allocation,
   input preparation, or handle transfer receive a generation-bound
   preparation grant for the Python actor to submit their completed descriptors.
2. The native dispatcher selects an eligible worker and device under the
   current credits. Its private actor channel carries the command and the
   lease's run, parent, task, worker, worker epoch, PID, and generation. The
   actor installs the task and acknowledges it before sending completion.
3. Native code validates the receipt against the selected channel and lease
   before updating the ledger. A compute release or a final arriving first
   returns the compute slot once. A final receipt makes the scientific result
   available to Python. After every task has a validated final receipt, the
   parent can leave the preparation window and its publication ticket becomes
   available to the consumer.
4. Python assembles the results and tracks the output consumers' lifetime. The
   sweep publisher provides a completion future, which the adapter services
   when it is ready. Native credits for retained parent storage remain live
   through the corresponding backing and aliases. The adapter reports observed
   retirement to the native owner using the parent generation and returns the
   ticket after publication and ownership settle.

A sweep child that hydrates a sealed native prediction uses the same parent
ticket and byte ledger for that workspace. The child adopts the hydrated
arrays and signals readiness before the parent enters publication.

The native loop wakes on ready submissions, accepted receipts, and device or
capacity changes. Already-prepared work can move while the Python result
consumer is occupied as long as worker, device, and credit limits permit.
New scientific input preparation and output completion still require their
Python producers and consumers.

The same native device ledger checks detector occupancy, auxiliary work,
projection claims and overlap, and SAM residency. Python probes CUDA and
exchanges generation-bound stage handles through `native_device_stages`.
Shared-device reuse requires the appropriate worker retirement and residency
proofs.

## SAM restoration grants

Restoring a deferred SAM parent uses a zero-task grant in the same native
byte and ticket ledger as detector work. The stager binds the exact backing
lease and byte cost to the grant's parent generation before the restore actor
starts. The actor's `Ready` receipt confirms that its inputs can be used.
`Retired` requires the same lease to be released, its mapping owners to reach
zero, and its registry entry to be gone. Native credit returns after both
receipts settle, including when the last mapping closes before `Ready` is
processed. A ready parent with an existing dense lease continues under that
lease.

An eligible oversized restore can request a singleton grant after passing the
physical memory and SAM startup checks. The native owner grants that request
only when the ledger is empty and the restore needs no reserved lane headroom.
It charges the full backing size and pauses further grants while outstanding
bytes exceed the normal hard limit. After detector drain, the native owner
repartitions capacity atomically only when the proposed partition can hold
all live charges and outstanding host declarations. Existing tickets and
charges carry across the accepted change.

Projection retirement pressure arms when adopted dense backings reach 75%
of configured dense capacity. It disarms at or below 62.5%. The scheduler
counts detector backings after scientific adoption and SAM backings after
validated readiness. File-result and seam escrow stay in ordinary byte
accounting.

## Detector backing handles

On Linux, the adapter describes registered memfd-backed arrays by backing
identity and access role. Native code pins those descriptors and sends them
with the prepared command over the worker socket using `SCM_RIGHTS`. Sources
and prefetched inputs use read-only handles. Result backings use read/write
handles. The worker verifies the identity and access mode before the install
receipt, then exposes the descriptors through its local procfs paths for the
scientific code to map.

The native owner uses the install receipt to record a worker's process-lifetime
source handles. Prefetch and result handles remain task-local. A valid install
receipt releases the native task's descriptor pins. Dense backing credit
follows the separate backing retirement protocol. Named backing files continue
to travel as paths on the native channel, including Windows runs.

## Diagnosing a stopped run

Preflight errors identify unsupported detector settings or extension build and
load failures. A malformed receipt, wrong lease, worker exit with outstanding
work, or task failure stops the run. The pipeline reports the original fault
before joining its executors. Correct the cause and rerun.

In runtime telemetry, `scheduler.native` records ready and active tasks,
blocked and preparing parents, byte and ticket occupancy, dispatch and final
counts, and the device stage claim count and epoch. `scheduler.wait_activity`
records the corresponding Python pipeline activity. During an intentional
backlog, compute can advance while publication still holds bytes. Native
dispatch waits once worker, device, or credit limits are exhausted.

The [native core tests](../tests/test_prepared_scheduler.py) exercise the real
extension, and the [lifecycle tests](../tests/test_native_scheduler_lifecycle.py)
exercise receipt rejection, worker exit, and dispatch during delayed Python
consumption. The [SAM staging tests](../tests/test_native_sam_staging.py)
use synthetic CPU cases to check admission and alias-safe restoration.
See the [architecture](../ARCHITECTURE.md#tta-inference-to-source-union) for
the surrounding scientific stages.
