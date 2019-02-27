NUM_CATEGORIES = 91  # Total number of categories in Coco dataset
NORMALIZE_CLASS_WEIGHTS = True
DO_KFOLD_CROSSVAL = False
SINGLE_CATEGORIES = False
SINGLE_CATEGORY = 1
SAVE_MODEL = True

dataDir = 'coco'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
