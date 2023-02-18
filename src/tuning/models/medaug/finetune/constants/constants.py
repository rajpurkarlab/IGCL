"""Define constants to be used throughout the repository."""
from pathlib import Path

# Main directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path("/deep/group/cxr-transfer")

# Datasets
CHEXPERT = "chexpert"
CUSTOM = "custom"
CHEXPERT_SINGLE = "chexpert_single_special"
CXR14 = "cxr14"
SHENZHEN = "shenzhen_special"
RANZCR = "ranzcr_catheter"
INDIANA = "indiana"
JSRT  = "jsrt"
SIIM = "siim_acr_pneumothorax"
PEDIATRIC = "pediatric-pneumonia"
SHENZHEN = 'shenzhen-TB'
PULMONARY_EDEMA = 'pulmonary-edema'
MONTGOMERY = 'montgomery'

# Predict config constants
CFG_TASK2MODELS = "task2models"
CFG_AGG_METHOD = "aggregation_method"
CFG_CKPT_PATH = "ckpt_path"
CFG_IS_3CLASS = "is_3class"

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COL_PATH = "Path"
COL_STUDY = "Study"
COL_TASK = "Tasks"
COL_METRIC = "Metrics"
COL_VALUE = "Values"
TASKS = "tasks"
UNCERTAIN = -1
MISSING = -2

#RANZCR specific constants
RANZCR_DATASET_NAME = "ranzcr-clip-catheter-line-classification"
RANZCR_PARENT_DATA_DIR  = DATA_DIR / "datasets/"
RANZCR_SAVE_DIR = DATA_DIR / "models/"
RANZCR_DATA_DIR = RANZCR_PARENT_DATA_DIR / RANZCR_DATASET_NAME
RANZCR_TEST_DIR = 0
RANZCR_UNCERTAIN_DIR = 0
RANZCR_RAD_PATH = 0


RANZCR_COL = "StudyInstanceUID"
RANZCR_COLPATH  = "StudyInstanceUID"

RANZCR_MEAN = [0.4827, 0.4827, 0.4827]
RANZCR_STD = [0.055004, 0.055004, 0.055004]

RANZCR_TASKS = ["ETT - Abnormal","ETT - Borderline","ETT - Normal","NGT - Abnormal","NGT - Borderline","NGT - Incompletely Imaged","NGT - Normal","CVC - Abnormal","CVC - Borderline","CVC - Normal","Swan Ganz Catheter Present"]


#SIIM_ACR_Pneumothorax specific constants

SIIM_DATASET_NAME = "siim-acr-pneumothorax-segmentation"
SIIM_PARENT_DATA_DIR = DATA_DIR / "datasets/"
SIIM_SAVE_DIR = DATA_DIR / "models/"
SIIM_DATA_DIR = SIIM_PARENT_DATA_DIR / SIIM_DATASET_NAME / "metadata/"
SIIM_TEST_DIR = 0
SIIM_UNCERTAIN_DIR = 0
SIIM_RAD_PATH = 0

SIIM_COL = "file_paths"
SIIM_COLPATH  = "file_paths"

SIIM_MEAN  = [0.4888,0.4888,0.4888]
SIIM_STD = [0.0619,0.0619,0.0619]

SIIM_TASKS = ["result"]


#PediatricPneumonia specific constants

PEDIATRIC_DATASET_NAME = "pediatric-pneumonia"
PEDIATRIC_PARENT_DATA_DIR = DATA_DIR / "datasets/"
PEDIATRIC_SAVE_DIR = DATA_DIR / "models/"
PEDIATRIC_DATA_DIR = PEDIATRIC_PARENT_DATA_DIR / PEDIATRIC_DATASET_NAME / "metadata/"
PEDIATRIC_TEST_DIR = 0
PEDIATRIC_UNCERTAIN_DIR = 0
PEDIATRIC_RAD_PATH = 0

PEDIATRIC_COL = "file_paths"
PEDIATRIC_COLPATH  = "file_paths"

PEDIATRIC_MEAN  = [0.4825,0.4825,0.4825]
PEDIATRIC_STD = [0.05614,0.05614,0.05614]

PEDIATRIC_TASKS = ["result"]


#Shenzhen specific constants

SHENZHEN_DATASET_NAME = "shenzhen"
SHENZHEN_PARENT_DATA_DIR = DATA_DIR / "datasets/"
SHENZHEN_SAVE_DIR = DATA_DIR / "models/"
SHENZHEN_DATA_DIR = SHENZHEN_PARENT_DATA_DIR / SHENZHEN_DATASET_NAME / "metadata/"
SHENZHEN_TEST_DIR = 0
SHENZHEN_UNCERTAIN_DIR = 0
SHENZHEN_RAD_PATH = 0

SHENZHEN_COL = "file_paths"
SHENZHEN_COLPATH  = "file_paths"

SHENZHEN_MEAN  = [0.4615,0.4615,0.4615]
SHENZHEN_STD = [0.077621,0.077621,0.077621]

SHENZHEN_TASKS = ["result"]


#Pulmonary-Edema Severity specific constants

PULMONARY_EDEMA_DATASET_NAME = "pulmonary-edema"
PULMONARY_EDEMA_PARENT_DATA_DIR = DATA_DIR / "datasets/"
PULMONARY_EDEMA_SAVE_DIR = DATA_DIR / "models/"
PULMONARY_EDEMA_DATA_DIR = PULMONARY_EDEMA_PARENT_DATA_DIR / PULMONARY_EDEMA_DATASET_NAME / "metadata/"
PULMONARY_EDEMA_TEST_DIR = 0
PULMONARY_EDEMA_UNCERTAIN_DIR = 0
PULMONARY_EDEMA_RAD_PATH = 0

PULMONARY_EDEMA_COL = "file_paths_320"
PULMONARY_EDEMA_COLPATH  = "file_paths_320"

PULMONARY_EDEMA_MEAN  = [0.46925,0.46925,0.46925]
PULMONARY_EDEMA_STD = [0.092811,0.092811,0.092811]

PULMONARY_EDEMA_TASKS = ["edema_score"]


#MONTGOMERY

MONTGOMERY_DATASET_NAME = "montgomery"
MONTGOMERY_PARENT_DATA_DIR  = DATA_DIR / "datasets/"
MONTGOMERY_SAVE_DIR = DATA_DIR / "models/"
MONTGOMERY_DATA_DIR = MONTGOMERY_PARENT_DATA_DIR / MONTGOMERY_DATASET_NAME
MONTGOMERY_TEST_DIR = 0
MONTGOMERY_UNCERTAIN_DIR = 0
MONTGOMERY_RAD_PATH = 0


MONTGOMERY_COL = "file_paths_320"
MONTGOMERY_COLPATH  = "file_paths_320"

MONTGOMERY_MEAN = [0.4356, 0.4356, 0.4356]
MONTGOMERY_STD = [0.112495, 0.112495, 0.112495]

MONTGOMERY_TASKS = ["result"]


#Indiana specific constants
INDIANA_DATASET_NAME = "indiana"
INDIANA_PARENT_DATA_DIR  = DATA_DIR / "datasets/"
INDIANA_SAVE_DIR = DATA_DIR / "models/"
INDIANA_DATA_DIR = INDIANA_PARENT_DATA_DIR / INDIANA_DATASET_NAME
INDIANA_TEST_DIR = 0
INDIANA_UNCERTAIN_DIR = 0
INDIANA_RAD_PATH = 0


INDIANA_COL = "file_paths_320"
INDIANA_COLPATH  = "file_paths_320"

INDIANA_MEAN = [0.481579, 0.481579, 0.481579]
INDIANA_STD = [0.069162, 0.069162, 0.069162]

INDIANA_TASKS = ["result"]



#Indiana specific constants
JSRT_DATASET_NAME = "jsrt"
JSRT_PARENT_DATA_DIR  = DATA_DIR / "datasets/"
JSRT_SAVE_DIR = DATA_DIR / "models/"
JSRT_DATA_DIR = JSRT_PARENT_DATA_DIR / JSRT_DATASET_NAME
JSRT_TEST_DIR = 0
JSRT_UNCERTAIN_DIR = 0
JSRT_RAD_PATH = 0


JSRT_COL = "file_paths_320"
v_COLPATH  = "file_paths_320"

JSRT_MEAN = [0.23981, 0.23981, 0.23981]
JSRT_STD = [0.04112, 0.04112, 0.04112]

JSRT_TASKS = ["result"]



# CheXpert specific constants
CHEXPERT_DATASET_NAME = "SSL-methods/MedAug/label_fractions/"
CHEXPERT_PARENT_DATA_DIR = DATA_DIR 
CHEXPERT_SAVE_DIR = CHEXPERT_PARENT_DATA_DIR / "models/"
CHEXPERT_DATA_DIR = CHEXPERT_PARENT_DATA_DIR / CHEXPERT_DATASET_NAME

CHEXPERT_TEST_DIR = Path("/deep/group/data")/"moco/chexpert-proper-test-random-v2/moving_logs"
# CHEXPERT_TEST_DIR = CHEXPERT_PARENT_DATA_DIR / "CodaLab"
CHEXPERT_UNCERTAIN_DIR = CHEXPERT_PARENT_DATA_DIR / "Uncertainty"
CHEXPERT_RAD_PATH = CHEXPERT_PARENT_DATA_DIR / "rad_perf_test.csv"
CHEXPERT_MEAN = [.5020, .5020, .5020]
CHEXPERT_STD = [.085585, .085585, .085585]
CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"
                  ]
CHEXPERT_SINGLE_TASKS = ["No Finding",
                         "Pleural Effusion",
                        ]

# CHEXPERT_SINGLE_TASKS = ["No Finding",
#                          "Edema",
#                         ]                        

CHEXPERT_COMPETITION_TASKS = ["Atelectasis",
                              "Cardiomegaly",
                              "Consolidation",
                              "Edema"
                              ]
CHEXPERT_COMPETITION_SINGLE_TASKS = ["Pleural Effusion"]
# CHEXPERT_COMPETITION_SINGLE_TASKS = ["Edema"]

SHENZHEN_TASKS = ['result']

# CXR14 specific constants
CXR14_DATA_DIR = DATA_DIR / CXR14
CXR14_TASKS = ["Cardiomegaly",
               "Emphysema",
               "Pleural Effusion",
               "Hernia",
               "Infiltration",
               "Mass",
               "Nodule",
               "Atelectasis",
               "Pneumothorax",
               "Pleural Thickening",
               "Pneumonia",
               "Fibrosis",
               "Edema",
               "Consolidation"]
CALIBRATION_FILE = "calibration_params.json"

DATASET2TASKS = {CHEXPERT: CHEXPERT_TASKS,
                 CUSTOM: CHEXPERT_TASKS,
                 CHEXPERT_SINGLE: CHEXPERT_TASKS,
                 CXR14: CXR14_TASKS,
                 RANZCR : RANZCR_TASKS,
                 SIIM : SIIM_TASKS,
                 SHENZHEN: SHENZHEN_TASKS,
                 MONTGOMERY: MONTGOMERY_TASKS,
                 INDIANA : INDIANA_TASKS,
                 JSRT : JSRT_TASKS,
                 PULMONARY_EDEMA: PULMONARY_EDEMA_TASKS}

EVAL_METRIC2TASKS = {'chexpert-log_loss': CHEXPERT_TASKS,
                     'cxr14-log_loss': CXR14_TASKS,
                     'shenzhen-AUROC': SHENZHEN_TASKS,
                     'chexpert-competition-log_loss': CHEXPERT_COMPETITION_TASKS,
                     'chexpert-competition-AUROC': CHEXPERT_COMPETITION_TASKS,
                     'ranzcr-competition-AUROC' : RANZCR_TASKS,
                     'siim-competition-AUROC' : SIIM_TASKS,
                     'pediatric-pneumonia-competition-AUROC': PEDIATRIC_TASKS,
                     'shenzhen-tb-competition-AUROC': SHENZHEN_TASKS,
                     'indiana-competition-AUROC' : INDIANA_TASKS,
                     'montgomery-AUROC':MONTGOMERY_TASKS,
                     'jsrt-AUROC':JSRT_TASKS,
                     'pulmonary-edema-AUROC':PULMONARY_EDEMA_TASKS,
                     'chexpert-competition-single-AUROC': CHEXPERT_COMPETITION_SINGLE_TASKS}

NamedTasks = {'chexpert': CHEXPERT_TASKS,
        'chexpert-competition': CHEXPERT_COMPETITION_TASKS,
        'pleural-effusion': ["Pleural Effusion"],
        'edema': ["Edema"],
        'consolidation': ["Consolidation"],
        'cardiomegaly': ["Cardiomegaly"],
        'atelectasis': ["Atelectasis"],
        'ranzcr':RANZCR_TASKS,
        'shenzhen':SHENZHEN_TASKS,
        'pediatric' : PEDIATRIC_TASKS,
        'indiana' : INDIANA_TASKS,
        'jsrt' : JSRT_TASKS,
        'montgomery' : MONTGOMERY_TASKS,
        'pulmonary-edema' : PULMONARY_EDEMA_TASKS,
        'siim':SIIM_TASKS
        }
