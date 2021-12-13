
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

df = pd.read_csv(r'C:\Users\Administrateur\Desktop\Football Dataset\datafm20.csv', sep=",")
df.drop("Unnamed: 0", axis=1, inplace=True)
columns_name = 'Wor : Work Rate,Vis : Vision, Thr : Throwing,Tec : Technique,Tea : Teamwork,Tck : Tackling,Str : Strength,Sta : Stamina,TRO : Rushing Out,Ref : Reflexes,Pun : Punching,Pos : Positioning,Pen : Penalty,Pas : Passing,Pac : Pace,1v1 : 1v1, OtB : Off the Ball,Nat : Natural Fitness,Mar : Marking,LTh : Long Throws,Lon : Long Shots,Ldr : Leadership,Kic : Kicking,Jum : Jumping Reach,Hea : Heading,Han : Handling,Fre : Free Kick,Fla : Flair,Fir : First Touch,Fin : Finishing,Ecc : Eccentricity,Dri : Dribbling,Det : Determination,Dec : Decisions,Cro : Crossing,Cor : Corners,Cnt : Concentration,Cmp : Composure,Com : Communication,Cmd : Command of Area,Bra : Bravery,Bal : Balance,Ant : Anticipation,Agi : Agility,Agg : Aggression,Aer : Aerial Reach,Acc : Acceleration'
columns_dict = {}
for i in range(0, len(columns_name.split(","))):
    columns_dict[columns_name.split(",")[i].strip()[:3]] = columns_name.split(",")[i][6:].strip()
df = df.rename(columns_dict, axis=1)
df = df.rename({"L Th" : "Long Throws"}, axis=1)


foot_dict = {"Left Only" : "Left", "Right Only" : "Right"}
df["Preferred Foot"] = df["Preferred Foot"].apply(lambda x: foot_dict[x] if x in foot_dict.keys() else x)
df = pd.concat([df, df["Preferred Foot"].str.get_dummies()], axis=1)

best_pos_dict = {"WB (R)" : "D (R)", "WB (L)" : "D (L)", "AM (L)" : "M (L)", "AM (R)" : "M (R)"}
df["Best Pos"] = df["Best Pos"].apply(lambda x: best_pos_dict[x] if x in best_pos_dict.keys() else x)

role_dict = {'W' : 'Winger', 'CD' : 'Central Defender', 'FB' : 'Full Back', 'P' : 'Poacher', 'CM' : 'Central Midfielder', 'SK' : 'Sweeper Keeper', 'G' : 'Goalkeeper', 'IW' : 'Inverted Winger',
 'AP' : 'Advanced Playmaker', 'BWM' : 'Ball Winning Midfielder', 'AF' : 'Advanced Forward', 'WB' : 'Wing Back', 'NCB' : 'No-Nonsense Center Back', 'DLP' : 'Deep Lying Playmaker',
 'PF' : 'Pressing Forward' , 'TM' : 'Target Man', 'MEZ' : 'Mezzala', 'AM' : 'Attacking Midfielder',
 'NFB' : 'No-Nonsense Full Back', 'A' : 'Anchorman', 'SS' : 'Shadow Striker', 'CAR' : 'Carrilero',
 'Unknown' : 'Unknown', 'DM' : 'Defensive Midfielder', 'BBM' : 'Box to Box', 'DW' : 'Defensive Winger',
 'BPD' : 'Ball Playing Defender', 'IWB' : 'Inverted Wing Back', 'CWB' : 'Complete Wing Back', 'F9' : 'F9', 'VOL' : 'Segundo Volante', 'DLF' : 'Deep Lying Forward',
 'HB' : 'Half Back', 'EG' : 'Enganche', 'IF' : 'Inside Forward', 'WTM' : 'Wide Target Man',
 'T' : 'Trequartista', 'RGA' : 'Regista', 'RPM' : 'Roaming Playmaker', 'CF' : 'Complete Forward',
 'WP' : 'Wide Playmaker', 'L' : 'Libero', 'RMD' : 'Raumdeuter', 'WM' : 'Wide Midfielder'}

df["Best Role"] = df["Best Role"].apply(lambda x: role_dict[x] if x in role_dict.keys() else x)


# only for field players
only_field = ["Tackling", "Penalty", "Marking", "Long Throws", "Long Shots", "Heading", "Finishing", "Dribbling",
             "Crossing", "Corners"]
# df pour gardien et pour joueur
df_gk = df[df["Position"] == "GK"].copy()
df_field = df[df["Position"] != "GK"].copy()
df_field.drop(["Throwing","Rushing Out","Reflexes","Punching","1v1","Kicking","Handling","Eccentricity", 
               "Communication","Command of Area","Aerial Reach"], axis=1, inplace=True)
df_gk.drop(only_field, axis=1, inplace=True)


X = df_field.drop(["Name", "Position", "Club",'Division', 'Based', 'Nation', 'Height',
       'Weight', 'Age', 'Preferred Foot', 'Best Pos', 'Best Role', 'Value',
       'Wage', 'CA', 'PA' ], axis=1)
y = df_field[["Best Pos", "Best Role"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

minmax_sc = MinMaxScaler()
X_train_mms = minmax_sc.fit_transform(X_train)
X_test_mms = minmax_sc.transform(X_test)


svc = SVC(gamma="scale")
model = MultiOutputClassifier(estimator=svc)

MultiOutputClassifier(estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                                    class_weight=None, coef0=0.0,
                                    decision_function_shape='ovr', degree=3,
                                    gamma='scale', kernel='rbf', max_iter=-1,
                                    probability=False, random_state=None,
                                    shrinking=True, tol=0.001, verbose=False),n_jobs=None) 

model.fit(X_train, y_train)
