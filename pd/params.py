
import sys 


BATCH_SIZE = 20000

## log 
PerfThreshold = 0.78
logBestIndiv = 20


# dirs
DATADIR = None
OUTDIR = None
if sys.platform == "darwin":
    DATADIR = "/Users/nimamanaf/Desktop/kaggle/pd/data/"
    OUTDIR = "/Users/nimamanaf/Desktop/kaggle/pd/data/out/"
elif sys.platform == 'win32':
    DATADIR = "C:\\Users\\20204069\\Desktop\\Kaggle\\pd\\data\\"
    OUTDIR = "C:\\Users\\20204069\\Desktop\\Kaggle\\pd\\data\\out\\"

elif sys.platform == 'linux':
	DATADIR = "/home/tue/20204069/pd/data/"
	OUTDIR = "/home/tue/20204069/pd/data/out/"
    #DATADIR = "/home/nimamd/pd/data/"
    #OUTDIR = "/home/nimamd/pd/data/out/"

# data
dataCols = ['customer_ID', 'S_2', 'P_2', 'D_39', 'B_1', 'B_2', 'R_1', 'S_3', 'D_41','B_3',
 	'D_42', 'D_43','D_44', 'B_4', 'D_45', 'B_5', 'R_2', 'D_46', 'D_47', 'D_48', 'D_49',
 	'B_6', 'B_7', 'B_8', 'D_50', 'D_51', 'B_9', 'R_3', 'D_52', 'P_3', 'B_10', 'D_53', 'S_5',
 	'B_11', 'S_6', 'D_54', 'R_4', 'S_7', 'B_12', 'S_8', 'D_55', 'D_56', 'B_13', 'R_5',
 	'D_58', 'S_9', 'B_14', 'D_59', 'D_60', 'D_61', 'B_15', 'S_11', 'D_62', 'D_63', 'D_64', 'D_65',
 	'B_16',  'B_17', 'B_18', 'B_19', 'D_66', 'B_20', 'D_68', 'S_12', 'R_6', 'S_13', 'B_21',
 	'D_69', 'B_22', 'D_70', 'D_71', 'D_72', 'S_15', 'B_23', 'D_73', 'P_4', 'D_74','D_75','D_76','B_24',
	'R_7', 'D_77','B_25', 'B_26', 'D_78', 'D_79', 'R_8', 'R_9', 'S_16', 'D_80', 'R_10','R_11', 'B_27', 'D_81',
 	'D_82','S_17', 'R_12', 'B_28', 'R_13', 'D_83', 'R_14', 'R_15', 'D_84', 'R_16', 'B_29', 'B_30', 'S_18',
	'D_86','D_87', 'R_17', 'R_18', 'D_88', 'B_31', 'S_19', 'R_19', 'B_32', 'S_20', 'R_20', 'R_21',
	'B_33', 'D_89', 'R_22', 'R_23', 'D_91', 'D_92', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'S_22',
	'S_23','S_24', 'S_25', 'S_26', 'D_102', 'D_103', 'D_104', 'D_105', 'D_106', 'D_107', 'B_36',
 	'B_37', 'R_26', 'R_27', 'B_38','D_108','D_109', 'D_110', 'D_111', 'B_39', 'D_112', 'B_40',
 	'S_27', 'D_113', 'D_114', 'D_115', 'D_116', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121',
 	'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_127', 'D_128', 'D_129', 'B_41', 'B_42', 'D_130', 'D_131','D_132',
 	'D_133','R_28', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142', 'D_143', 'D_144',
	'D_145']
CATCOLS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
ContCols = [col for col in dataCols if col not in CATCOLS + ["customer_ID", "S_2", "target"]]
MostNaNCols = ['D_42', 'D_50', 'D_53', 'D_73', 'D_76', 'B_29', 'D_88', 'D_110', 'B_39', 'B_42', 'D_132', 'D_134', 'D_142']


## A set of cols created by me looking into thre distribution of the customers who deafult and customers who don't
ManCols = ["P_2", "B_1", "B_2", "R_1", "S_3", "D_41", "B_3", 
	"D_42", "D_43", "D_44", "B_4", "D_45", "B_5", "R_2", 
	"D_46", "D_47", "D_48", "D_49", "B_6", "B_7", "B_8", 
	"D_50", "D_51", 'B_9', "R_3", "D_52", "P_3", "B_10",
	"D_53", "S_5", "B_11", "S_6", "R_4", "S_7", "B_12", 
	"S_8", "D_55", "D_56", "D_58", "D_59", "D_60", "D_61",
	"S_11", "D_62", "D_64", "B_16", "B_17", "B_18", "B_19", 
	"B_20", "D_68", "B_22", "D_70", "S_15", "B_23", "D_74", 
	"D_75", "D_76", "D_78", "D_79", "B_28", "B_30", "B_33", 
	"S_22", "S_23", "S_24", "B_37", "R_26", "B_38", "B_40", 
	"D_124", "D_128", "D_129"]
    

## Learning 