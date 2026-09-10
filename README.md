# Session-based-Music-queue-Reordering-System-Using-Skip-Behaviour-as-Implicit-Feedback

## Repository Layout 

| Directory | Contents |
|---|---|
| `Data Scripts/` | The five-stage data pipeline, orchestrated by `data_processing_main.py` |
| `Model Scripts/` | `Model_lib.py` (datasets, both architectures, evaluation metrics, training loop), `Loss_functions.py`, the four model entry points and their tuning scripts |
| `Report Scripts/` | `evaluate.py`, `context_analysis.py`, `demo_reorder.py` and results tables used in Chapter 6 |
| `Models/` | Grid-search results, the selected configuration per arm, the trained checkpoints and `final_results.csv` |
| `Info/` | Volunteer information sheet and blank consent form as issued, ethics application, Project Proposal |
| `run_lstm.sh`, `run_sasrec.sh`, `submit_all.sh` | SLURM job-array submission scripts | 
## Order of execution of code

```bash
python3 “Data Scripts”/data_processing_main.py
.submit_all.sh bce
.submit_all.sh pwts
.submit_all.sh drrl
python3 “Model Scripts”/RNN.py --loss pwts
python3 “Model Scripts”/RNN.py --loss drrl
python3 “Model Scripts”/RNN.py --loss bce
python3 “Model Scripts”/SASRec.py --loss bce
python3 “Model Scripts”/SASRec.py --loss pwts
python3 “Model Scripts”/SASRec.py --loss drrl
python3 “Model Scripts”/Log_regression.py
python3 “Model Scripts”/ExtraTreesClassifier.py
python3 “Report Scripts”/evaluate.py
python3 “Report Scripts”/context_analysis.py
python3 “Report Scripts” /demo_reorder.py (3 times)
```

## Environment   
Python 3.14.3 with PyTorch 2.12.0, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4 and PyArrow 24.0.0. Tuning and training were run as SLURM job arrays on the school’s GPU cluster; initially the author attempted to run on Google colab, however this was costly, this is why there is still ‘cuda’ present in some of the scripts; every script can run on a single machine, hence the option of using ‘mps’ or ‘cpu’ but the 1,080 configuration search is too much workload for a laptop.

## Data 
The volunteers’ streaming histories and the processed datasets are excluded by .gitignore, in line with the ethics application (Appendix D): Raw Data/, Volunteer Data/ and all .parquet and .json files. All scripts to carry out the listening analysis for the volunteers have also been gitignored as they are not relevant to this research.

## Third party code
The self-attentive model in Model_lib.py is adapted from SASRec.pytorch Huang (2020), licensed under Apache 2.0; PointWiseFeedForward and the attention-block loop are unchanged from that source, and the modifications are described in Section 3.2.1. The DrRLLoss class in Loss_funcitons.py impletments the objective of Zhang et al. (2025) following their reference implementation, and shares no code with it.
