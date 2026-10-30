# MethRidgeQC

MethRidgeQC assesses strand-specific DNA methylation measurements using cytosine counts and Ridge regression. It checks CpG positions against a reference genome, estimates count-based reliability risk, and optionally assigns reliability grades. Lower risk indicates greater estimated reliability. Grades run from H to MH, ML, and L; U marks an unclassified record.

The package can evaluate existing QC results against independent methylation reference truth without refitting the model. It does not correct methylation levels, merge complementary strands, or certify measurement accuracy.

Version: 0.1.0. License: MIT (see [LICENSE](LICENSE)).

We plan to publish MethRidgeQC packages on Bioconda and PyPI. Until those releases are available, install from source using Pixi or uv as described below.

## Contents

- [Installation with Pixi](#installation-with-pixi)
- [Installation with uv](#installation-with-uv)
- [Command-line usage](#command-line-usage)
- [Python API](#python-api)
- [Method and interpretation](#method-and-interpretation)
- [Input requirements](#input-requirements)
- [Output formats and contents](#output-formats-and-contents)

## Installation with Pixi

The source distribution provides a Pixi manifest and lockfile for Linux x86-64. Both environments use Python 3.13; the package requires Python `>=3.13,<3.14`.

Install [Pixi](https://pixi.sh/) using its official instructions. The commands below install MethRidgeQC from a local source checkout.

### CPU environment

```sh
git clone https://github.com/YuanfengZhang/MethRidgeQC.git
# OR uncompress the zipped archive file.
cd MethRidgeQC

pixi install -e default --locked
pixi run -e default --locked methridgeqc --version
pixi run -e default --locked methridgeqc qc --help
```

The `default` environment runs on CPU and does not require NVIDIA hardware. Its numerical dependencies include NumPy, SciPy, Polars, PyArrow, and scikit-learn.

Run the CPU test suite:

```sh
pixi run -e default --locked test
```

Tests automatically create `temp/pytest/` under the project root; no manual temporary-directory setup is needed.

### GPU environment

The optional `gpu` environment adds cuML 26.10, RMM 26.10, and CuPy, with CUDA 13 packages. A supported NVIDIA GPU and a host driver compatible with the CUDA 13 runtime resolved in `pixi.lock` are required. Pixi does not install the host NVIDIA driver. The CUDA version reported by `nvidia-smi` describes driver capability, not an installed CUDA Toolkit.

From the source root:

```sh
pixi install -e gpu --locked
pixi run -e gpu --locked methridgeqc --version
pixi run -e gpu --locked test
```

Select GPU execution explicitly with `--gpu`, or `gpu=True` in Python. Running inside the GPU environment alone does not select the GPU backend. If cuML, CuPy, or CUDA cannot be loaded, an explicit GPU request fails rather than silently falling back to CPU.

The GPU test task sets `CUDA_PATH` to the environment's CUDA target directory. If an ordinary GPU run reports missing CUDA headers, set `CUDA_PATH` to the matching CUDA installation for that environment.

Use `--locked` to keep the resolved dependency versions.

## Installation with uv

You can also install from a local source directory with [uv](https://docs.astral.sh/uv/). Install uv using its official instructions, then create a Python 3.13 environment as shown below.

### CPU environment

From an existing local MethRidgeQC source directory containing `pyproject.toml`:

```sh
cd MethRidgeQC
uv venv --python 3.13 .venv
. .venv/bin/activate
uv pip install .

methridgeqc --version
methridgeqc qc --help
```

The default installation is CPU-only: it does not request cuML, CuPy or CUDA packages. If Python 3.13 is not available locally, uv can download a compatible interpreter.

For CPU development and tests:

```sh
uv pip install -e '.[test]'
python -m pytest
```

### GPU environment

On Linux x86-64, use the `gpu` extra to install cuML 26.10, RMM 26.10, and CuPy for CUDA 13. Use a separate environment rather than mixing CPU/GPU package variants:

```sh
cd MethRidgeQC
uv venv --python 3.13 .venv-gpu
. .venv-gpu/bin/activate
uv pip install '.[gpu]'

methridgeqc --version
methridgeqc qc --help
python -c "import cupy, cuml; print('cuML:', cuml.__version__); print('GPU devices:', cupy.cuda.runtime.getDeviceCount())"
```

The GPU extra requests CUDA 13 user-space libraries through its dependencies; it does not install or upgrade the host NVIDIA driver. A supported NVIDIA GPU and a driver compatible with the resolved CUDA 13 runtime are required. `nvidia-smi` reports driver capability, not the installed Toolkit version. Do not mix `cupy`, `cupy-cuda12x` and `cupy-cuda13x` in one environment. CUDA 13.x supports minor-version compatibility with driver branch 580 or newer, but PTX JIT and newer features can require a newer driver; consult [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) and verify actual GPU computation, not only device discovery.

For GPU development and tests:

```sh
uv pip install -e '.[gpu,test]'
METHRIDGEQC_TEST_GPU=1 python -m pytest
```

Select GPU computation explicitly with `methridgeqc qc --gpu ...`, or `gpu=True` in the Python API. Installing the GPU extra alone does not switch the default backend. An explicit GPU request fails if cuML, CuPy or CUDA is unavailable; it never silently falls back to CPU. Evaluation has no GPU option.

The Pixi-specific `$CONDA_PREFIX/targets/x86_64-linux` path does not apply to uv environments. If CUDA headers cannot be found, inspect the CUDA Toolkit installed for that environment and set `CUDA_PATH` to its actual root, rather than copying a Conda path.

Both installations use the same MethRidgeQC package; CPU/GPU selection is an optional dependency group, not a separate package release. uv resolves dependencies from `pyproject.toml`, not `pixi.lock`, so its environments are not exact reproductions of the locked Pixi environments.

With the uv environment activated, run `methridgeqc ...` directly: remove `pixi run -e default --locked` from the CPU command examples below. Likewise, run Python scripts with `python script.py`. Use `deactivate` to leave the environment.

## Command-line usage

Run the examples from the source root. Create output and workspace directories before execution. Explicitly setting `--work-dir` keeps intermediate files under the project rather than in the system temporary directory.

### Example

The bundled files contain 1,144,919 count rows and 690,864 reference rows in complete chromosome 22 subsets of BS1_D6_1 counts, D6 reference truth, and soft-masked GRCh38 sequence with its FAI index. See [example/README.md](example/README.md) for provenance, schemas, and limitations. Coordinates remain genome-relative; beta values use percent units. This single-chromosome example is not a genome-wide performance benchmark. Synthetic boundary fixtures remain in unit tests.

```sh
mkdir -p try/quickstart try/temp

pixi run -e default --locked methridgeqc qc \
  --input example/BS1_D6_1.parquet.lz4 \
  --reference example/GRCh38_chr22.fa \
  --output try/quickstart/qc.parquet.lz4 \
  --output-format parquet.lz4 \
  --score --classify \
  --model-out try/quickstart/ridge.json \
  --work-dir try/temp

pixi run -e default --locked methridgeqc eval \
  --input try/quickstart/qc.parquet.lz4 \
  --reference example/GRCh38_chr22.fa \
  --truth example/D6_reference.parquet.lz4 \
  --truth-scale percent \
  --output try/quickstart/metrics.tsv \
  --bootstrap 10 \
  --work-dir try/temp
```

### Validate and filter without fitting a model

```sh
pixi run -e default --locked methridgeqc qc \
  --input counts.tsv --reference genome.fa \
  --output filtered.bedgraph \
  --work-dir try/temp
```

Without `--score` or `--classify`, QC validates the input, removes non-CpG positions, and adds validation annotations. No Ridge model is fitted.

### Fit, score, and classify on CPU

```sh
pixi run -e default --locked methridgeqc qc \
  --input counts.parquet --reference genome.fa \
  --output qc.parquet.lz4 --output-format parquet.lz4 \
  --score --classify --threads 4 \
  --model-out ridge.json \
  --work-dir try/temp
```

### Fit, score, and classify on GPU

```sh
pixi run -e gpu --locked methridgeqc qc \
  --input counts.parquet --reference genome.fa \
  --output qc.parquet.lz4 --output-format parquet.lz4 \
  --score --classify --gpu \
  --model-out ridge-gpu.json \
  --work-dir try/temp
```

### Apply a frozen model to another sample

```sh
pixi run -e default --locked methridgeqc qc \
  --input next.tsv --reference genome.fa \
  --output next.bedgraph.gz --output-format bedgraph.gz \
  --score --classify --model-in ridge.json \
  --work-dir try/temp
```

A frozen model reuses preprocessing, regression parameters, quantization, and grade boundaries without refitting. CPU can read models fitted on GPU. Numerical agreement across backends should be assessed with tolerances, not byte equality.

### Streams

Use `-` for standard input or standard output. TSV is the default stdin format; Parquet stdin requires `--input-format parquet` and is staged once in the workspace.

```sh
cat counts.tsv | pixi run -e default --locked methridgeqc qc \
  --input - --reference genome.fa \
  --output - --output-format bedgraph.gz --score \
  --work-dir try/temp > qc.bedgraph.gz
```

Data go to stdout when requested; diagnostics and the final JSON run summary go to stderr. Gzip and Parquet stdout are binary streams and must be redirected appropriately.

### Common options

| Option | Meaning / default |
| --- | --- |
| `--score` | Export `qc_risk` and `qc_risk_quantized`. |
| `--classify` | Export `qc_grade`; internal scoring does not force risk-column export. |
| `--output-format` | QC output: `bedgraph` (default), `bedgraph.gz`, or `parquet.lz4`. The filename does not select the output format. |
| `--input-format` | Explicit input parser: `tsv`, `parquet`, `bedgraph`, or `bedgraph.gz`. |
| `--chromosomes chr1 chr2` | Restrict processing to selected reference contigs; default: all input contigs compatible with the reference. |
| `--threads` | QC chromosome workers, integer `1` to `120`; default `1`. `auto` uses one chromosome worker without imposed numerical thread limits. GPU uses one chromosome worker. |
| `--max-train` | Deterministic training sample cap, `1` to `200000`; default `200000`. Does not cap the number of scored rows. |
| `--model-in` | Load a compatible frozen JSON model when scoring or classification is requested. |
| `--model-out` | Save a newly fitted model. Not a model-copy operation when `--model-in` is used. |
| `--seed` | Default `1729`; evaluation requires a non-negative integer. |
| `--work-dir` | Parent directory for temporary workspaces. Successful workspaces are removed; failed workspaces are retained for diagnosis. |
| `--truth-format` | Evaluation truth parser: `tsv` or `parquet`. |
| `--truth-scale` | Truth beta units: `percent` (default) or `fraction`; never inferred from values. |
| `--bootstrap` | Evaluation bootstrap replicates; default `300`. Use `0` for point estimates without confidence intervals. |
| `--log-level` | `DEBUG`, `INFO` (default), `WARNING`, or `ERROR`; accepted before or after the subcommand. Does not suppress the final JSON summary. |

Full command reference:

```sh
pixi run -e default --locked methridgeqc --help
pixi run -e default --locked methridgeqc qc --help
pixi run -e default --locked methridgeqc eval --help
```

The CLI returns `0` on success and `2` for handled input, runtime, resource, or filesystem errors. Processing is chromosome-wise; memory must accommodate a whole chromosome and its features. Increasing chromosome concurrency increases memory demand.

## Python API

The public entry points are `methridgeqc.qc.run` and `methridgeqc.eval.run`. Both are file-oriented APIs: they accept paths, write output files, return a dictionary of run statistics, and also emit that dictionary as a JSON line on stderr. They do not return the output table as a DataFrame.

Save this example as `run_qc.py` in the source root:

```python
from pathlib import Path
from typing import Any

from methridgeqc import qc
from methridgeqc import eval as evaluation

output_dir = Path('try/python-example')
workspace_dir = Path('try/temp')
output_dir.mkdir(parents=True, exist_ok=True)
workspace_dir.mkdir(parents=True, exist_ok=True)

qc_summary: dict[str, Any] = qc.run(
    input='example/BS1_D6_1.parquet.lz4',
    reference='example/GRCh38_chr22.fa',
    output=str(output_dir / 'qc.parquet.lz4'),
    output_format='parquet.lz4',
    score=True,
    classify=True,
    gpu=False,
    threads=1,
    max_train=200000,
    seed=1729,
    model_out=str(output_dir / 'ridge.json'),
    work_dir=str(workspace_dir),
)

eval_summary: dict[str, Any] = evaluation.run(
    input=str(output_dir / 'qc.parquet.lz4'),
    reference='example/GRCh38_chr22.fa',
    truth='example/D6_reference.parquet.lz4',
    output=str(output_dir / 'metrics.tsv'),
    truth_scale='percent',
    bootstrap=300,
    seed=1729,
    work_dir=str(workspace_dir),
)
```

Run it in the CPU environment:

```sh
pixi run -e default --locked python run_qc.py
```

For GPU execution, set `gpu=True` in `qc.run` and run the script in the `gpu` environment. Evaluation itself has no GPU option. To reuse a model, pass `model_in='ridge.json'` with `score=True` and/or `classify=True`. The Python equivalent of CLI `--threads auto` is `threads=None`. Both APIs also accept `input_format` and `chromosomes`; evaluation additionally accepts `truth_format`.

Invalid contracts raise `ValueError`; unavailable explicit GPU execution raises `RuntimeError`. Filesystem errors propagate as `OSError` subclasses. These exceptions are not converted to CLI exit codes when calling Python directly.

## Method and interpretation

MethRidgeQC derives eight features from the counts: depth, opposite-strand depth, raw methylated fraction, Jeffreys posterior interval width, posterior standard deviation, absolute strand difference, standardized strand difference, and an opposite-strand missingness indicator.

Training targets combine posterior count uncertainty with opposite-strand disagreement, when an informative opposite strand exists, using a methylation-fraction tolerance of `0.10`. The model uses median imputation, scaling, and Ridge regression. Spatial cross-validation selects regularization using a one-standard-error rule; single-block training falls back to a fixed regularization value. Training uses seeded NumPy random keys assigned in coordinate order within each contig and is capped deterministically, but prediction covers every eligible record. Bootstrap sampling also uses seeded NumPy random generators. Fixed inputs and seeds remain reproducible within this implementation; training samples and bootstrap intervals can differ from earlier versions that used hash-derived keys.

Risk quantization and four-grade boundaries are frozen from training data. Grade support requires sufficient risk variation and spatial representation; unsupported grading produces `U`, including for otherwise valid records. Grades are not fixed universal risk thresholds, and comparisons across independently fitted models require care. Neither risk nor grade is a substitute for external validation.

## Input requirements

### Reference genome

- Supply an uncompressed FASTA through `--reference` / `reference`.
- Contig names must match input names exactly, including prefixes such as `chr`.
- Coordinates and reference must use the same assembly.
- An existing adjacent `.fai` index is used when present and must match the FASTA. Otherwise, the package builds an index in memory without modifying the reference. Use consistently wrapped FASTA sequence lines.
- Strand is inferred from the reference: the C of a forward-reference `CG` is `+`; its adjacent G represents the complementary cytosine and is `-`.

### QC count table

Provide a tab-separated table with a header or a Parquet table with the same required column names. Column order may vary. Additional columns are preserved unless they conflict with reserved annotations.

| Column | Required content |
| --- | --- |
| `chrom` | Non-null reference contig name. |
| `start` | Finite, integer-valued, zero-based genomic position. |
| `end` | Exclusive end coordinate; must equal `start + 1`. |
| `m` | Methylated cytosine count: finite, non-negative integer. |
| `u` | Unmethylated cytosine count: finite, non-negative integer. |
| `depth` | Finite, non-negative integer; must equal `m + u`. |
| `beta` | Original methylation percentage: finite numeric value in `[0, 100]`. Not a fraction. |

The first four rows of `example/BS1_D6_1.parquet.lz4`, with columns reordered and literal tab delimiters:

```text
chrom	start	end	m	u	depth	beta
chr22	10510234	10510235	2	0	2	100.0
chr22	10510235	10510236	0	2	2	0.0
chr22	10510274	10510275	1	1	2	50.0
chr22	10510275	10510276	1	1	2	50.0
```

Each row represents one strand-specific cytosine, not a merged CpG dyad. The two adjacent positions of a reference CpG may both appear, with independent counts. Missing opposite-strand records are allowed. An observed opposite strand with zero depth is distinct from an absent row, but neither provides an informative opposite-strand posterior.

Validation rules:

- Counts are mandatory. A conventional four-column bedGraph cannot be used directly, and counts are never reconstructed from beta.
- The original beta is retained; it is not recomputed from the counts, and agreement with `100 * m / depth` is not required by the input validator. All count-based calculations use the counts.
- Unknown contigs, out-of-range or non-single-base coordinates, duplicate `(chrom, start)` keys, invalid beta values, and reserved annotation conflicts cause failure.
- Non-CpG positions are removed and counted in the run summary.
- CpG rows with invalid counts or zero depth are retained, marked invalid, and receive no risk score. If classification is requested, their grade is `U`.
- Raw QC input must not already contain `strand`, `canonical_start`, `qc_valid`, `qc_reason`, `qc_risk`, `qc_risk_quantized`, or `qc_grade`. Existing QC output belongs in `eval`, not as raw input for another `qc` run.
- Input need not be sorted. QC output is ordered lexicographically by contig name and numerically by `start` within each contig.

### Input encoding and format selection

- TSV requires tab delimiters and a header; `NA` and `null` are recognized as null tokens. Required columns must retain appropriate numeric types.
- Text lines starting with `#` and blank lines are skipped. A `#columns` line followed by tab-separated names supplies the header for extended bedGraph input.
- `.parquet` and `.parquet.lz4` suffixes select Parquet automatically. Other filenames default to TSV unless `--input-format` is supplied.
- Gzip-compressed text files are opened automatically when their filenames end in `.gz`. Gzip stdin is not automatically decompressed; decompress it before piping.
- Extended bedGraph output from this package is readable by `eval`; standard four-column bedGraph still lacks the required counts.
- `--input-format parquet` selects Parquet even for a nonstandard suffix or stdin. Parquet LZ4 compression is internal, not an external compression layer.

### Independent truth table for evaluation

Evaluation requires the seven-column count schema in its QC input, plus any existing risk and grade columns to be evaluated. It does not generate missing scores or grades. Existing strand annotations must agree with the reference. Present risks must be finite values in `[0, 1]`, and invalid-count or zero-depth rows must have null risks and grade `U` when those columns exist.

Truth is a separate TSV or Parquet table with these six required columns:

| Column | Required content |
| --- | --- |
| `chrom` | Reference contig name. |
| `start` | Zero-based single-base position. |
| `end` | `start + 1`. |
| `strand` | `+` or `-`, matching the reference-inferred CpG strand. |
| `type` | `CG` or `CpG` (case-sensitive). |
| `beta` | Truth methylation percentage `[0, 100]` for `percent`, or fraction `[0, 1]` for `fraction`. |

The first two rows of `example/D6_reference.parquet.lz4`, showing only the required columns:

```text
chrom	start	end	strand	type	beta
chr22	42293855	42293856	+	CpG	80.27
chr22	16704563	16704564	+	CpG	26.69
```

Select truth units explicitly with `--truth-scale` / `truth_scale`; the default is `percent`. Truth is aligned by contig, position, and reference-normalized strand. Duplicate reference-normalized truth keys, unknown contigs, and invalid coordinates cause failure. Malformed truth beta, strand, or type values are tracked as `invalid_record`; unmatched QC positions are `absent`. Neither is treated as a valid truth match.

## Output formats and contents

### QC table

Select serialization explicitly with `--output-format` / `output_format`:

| Format | Encoding and intended use |
| --- | --- |
| `bedgraph` | UTF-8 tab-separated extended bedGraph, with metadata and column-name comments. Default. |
| `bedgraph.gz` | The same extended bedGraph, gzip-compressed. |
| `parquet.lz4` | A Parquet file with internal LZ4 compression, typed columns, nulls, and schema metadata. Read directly with a Parquet reader; do not externally decompress it. |

Original fields retain their measurement values; this is not a guarantee of identical physical types or byte-for-byte Parquet preservation. Additional user columns are retained.

For extended bedGraph, the first seven columns are always:

```text
chrom  start  end  beta  m  u  depth
```

The actual delimiter is a tab. User-defined extra columns follow, then QC annotations. The `#columns` header is authoritative. The first four columns resemble bedGraph, but beta remains a percentage, and the file contains additional columns. Consumers requiring standard four-column bedGraph must explicitly select the first four data fields and handle the comment lines.

The text header begins with:

```text
#{"format": "MethRidgeQC_extended_bedgraph_v1", "beta_unit": "percent", "risk_direction": "lower_is_more_reliable"}
#columns	chrom	start	end	beta	m	u	depth	strand	canonical_start	qc_valid	qc_reason
```

The example header shows validation-only output without extra input columns. Requested scores and grades add their column names. Parquet stores the same three metadata key/value pairs in its schema, including the shared format identifier.

QC annotations:

| Column | Presence | Meaning |
| --- | --- | --- |
| `strand` | Always for retained nonempty QC records | Reference-inferred `+` or `-` strand. |
| `canonical_start` | Always for retained nonempty QC records | `start` on `+`; `start - 1` on `-`. Shared CpG coordinate for pairing and spatial blocks, not a merged measurement. |
| `qc_valid` | Always for retained nonempty QC records | Boolean: legal counts and positive depth. Does not guarantee grade support or external accuracy. |
| `qc_reason` | Always for retained nonempty QC records | Empty string for valid records without a further reason; `invalid_counts`, `zero_depth`, or `grade_unsupported:<reason>` as applicable. |
| `qc_risk` | With `--score` | Continuous count-derived Ridge risk in `[0, 1]`; lower is more reliable. Null for invalid or unscorable records. |
| `qc_risk_quantized` | With `--score` | Frozen training-bin risk representative in `[0, 1]`; at most 128 ordered representatives. Null when unavailable. |
| `qc_grade` | With `--classify` | `H`, `MH`, `ML`, `L`, or `U`. Unsupported grading and invalid records receive `U`. |

`--score` and `--classify` are independent. Classification alone exports no risk columns. Unsupported grading can coexist with finite risk scores and `qc_valid=true`. If no valid training records exist and no frozen model is supplied, requested scores are null and grades are `U`; no new model file is saved.

Text nulls are written as `NA`; Parquet uses native nulls. An entirely empty retained dataset is written with the base seven-column schema rather than the full annotation schema.

### Frozen model JSON

`--model-out` / `model_out` writes an indented JSON object when a new model is fitted. It contains:

- Imputation and scaling parameters, Ridge coefficients, intercept, selected regularization, and backend identity.
- Scientific protocol, feature order, input schema identifier, and tolerance `delta`.
- Cross-validation candidates and status, training-row count, training cap, seed, and deterministic sampling description.
- Frozen quantization edges, risk representatives, grading cuts, support status, and quantization diagnostics.

The model contains numerical parameters rather than a pickled estimator. Treat it as a versioned MethRidgeQC artifact: load it with `--model-in` / `model_in` rather than editing its fields manually. The model does not contain the QC result table or independent truth observations.

### Evaluation report TSV

`eval` writes a header-bearing TSV regardless of the output filename. It produces a long-format metric table, not a row-level truth-joined table. Missing numeric results are `NA`.

| Column | Meaning |
| --- | --- |
| `domain` | Evaluation domain; currently `all_selected_contigs`. |
| `group` | `all`, individual grades, `H+MH`, or a score representation such as `quantized`, `continuous`, or `ordinal`; diagnostic placeholder groups may also occur. |
| `threshold` | Absolute-error event threshold in methylation-fraction units, where applicable; otherwise `NA`. |
| `coverage` | Requested risk-based retention fraction, where applicable; otherwise `NA`. Distinct from rows whose metric is `coverage`. |
| `metric` | Metric or diagnostic name. |
| `estimate` | Point estimate. |
| `ci_lower`, `ci_upper` | Conditional 95% percentile bootstrap interval; `NA` if unavailable. |
| `n` | Supporting count: group size for coverage/truth-state rows, matched-record count for error metrics, or the relevant scored support for score diagnostics. Tie-based retention can produce fractional counts. |
| `status` | `ok` for finite estimates, or a reason such as `no_valid_truth_matches`, `empty_matches`, `single_class`, `insufficient_scored_coverage`, `missing_score_columns`, or `missing_grade_column`. |

Evaluation defines error as:

```text
error = m / depth - truth_fraction
```

The original input beta is not used in this error calculation. Metrics include:

- Coverage of all records and available grade groups, relative to all retained input CpG records.
- MAE and RMSE, calculated only from valid-count, positive-depth records with valid truth matches. Values are methylation fractions, not percentages.
- Event rate for strict absolute-error thresholds `> 0.05`, `> 0.10`, and `> 0.20` (5, 10, and 20 percentage points).
- Truth-state counts: `truth_valid`, `truth_invalid_record`, and `truth_absent`. These describe truth availability and validity; valid truth alone does not make an invalid-count QC row eligible for error metrics.
- AUROC and AUPRC, treating larger risk or worse ordinal grade as evidence of an error event. If both risk columns exist, quantized risk takes precedence over continuous risk. Ordinal grades are evaluated separately when available; `U` is not a ranked reliability grade.
- Risk-based retention metrics at requested coverage `0.10`, `0.25`, `0.50`, and `0.75`: MAE, RMSE, event rates, `achieved_coverage`, `boundary_risk`, and `boundary_tie_retained_fraction`. Boundary ties use expected fractional retention without using truth to break ties. Insufficient scored coverage is reported explicitly.

Bootstrap resampling uses chromosome-local 1 Mb spatial blocks, including unobserved blocks in the selected chromosome universe. Intervals are conditional on the frozen QC model; they do not include model-refitting uncertainty. With `bootstrap=0`, estimates remain available but interval columns are `NA`.

### Run summaries and diagnostics

Each successful command and API call emits a final machine-readable JSON line on stderr. Both public APIs return the same summary dictionary. Ordinary log lines may precede it.

Common summary fields include `input_rows`, `non_cpg`, `excluded_contig`, `invalid_counts`, `zero_depth`, and processed `chromosomes`. QC also records backend, training-row count, seed, training cap, model-reuse status, cross-validation status, and quantizer support. Evaluation records truth scale, bootstrap count, seed, reference path, and interval definition.

Successful intermediate workspaces are cleaned automatically. On failure, the workspace is retained and its location is logged to stderr. A failed run can leave partial output; do not treat that output as a completed result.
