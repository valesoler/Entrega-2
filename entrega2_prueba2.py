import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Cargar dataset
df = pd.read_csv("../dataset_entrega_2/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv")

# Exploración inicial
#print(df.head())
#print(df.info())
#print(df.describe(include='all'))

# Estilo general de los gráficos
sns.set(style="whitegrid", palette="muted")
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

import os

# Crear carpeta automáticamente si no existe
output_folder = "resultados_graficos"
os.makedirs(output_folder, exist_ok=True)

# Histograma
plt.figure(figsize=(8,5))
sns.histplot(df["Gene_A_Oncogene"], bins=30, kde=True, color=sns.color_palette("viridis", as_cmap=True)(0.7))
plt.title("Distribución de la expresión del gen oncogénico (Gene_A_Oncogene)")
plt.xlabel("Nivel de expresión")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"1_Distribucion_Gene_A_Oncogene.png"), dpi=300)
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x="Cell_Type", y="Gene_B_Immune", data=df, palette="Set2")
plt.title("Expresión del gen inmune (Gene_B_Immune) según tipo celular")
plt.xlabel("Tipo de célula")
plt.ylabel("Nivel de expresión")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "2_Boxplot_Gene_B_Immune_CellType.png"), dpi=300)
plt.show()

# Scatterplot
plt.figure(figsize=(7,5))
sns.scatterplot(x="Gene_C_Stromal", y="Gene_D_Therapy",
                hue="Disease_Status", data=df,
                palette="tab10", alpha=0.8)
plt.title("Relación entre expresión del gen de inflamación y el gen de terapia")
plt.xlabel("Gene_C_Stromal (nivel de expresión)")
plt.ylabel("Gene_D_Therapy (nivel de expresión)")
plt.legend(title="Estado de enfermedad")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "3_Scatter_Inflam_vs_Therapy.png"), dpi=300)
plt.show()

# Countplot
plt.figure(figsize=(8,5))
sns.countplot(x="Disease_Status", hue="Cell_Type", data=df, palette="Set2")
plt.title("Distribución de tipos celulares por estado de enfermedad")
plt.xlabel("Estado de enfermedad")
plt.ylabel("Número de muestras")
plt.legend(title="Tipo de célula")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "4_Countplot_CellType_DiseaseStatus.png"), dpi=300)
plt.show()

# Heatmap
genes = ["Gene_A_Oncogene", "Gene_B_Immune", "Gene_C_Stromal",
         "Gene_D_Therapy", "Pathway_Score_Inflam"]
plt.figure(figsize=(9,6))
corr = df[genes].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de correlación entre genes y vías de expresión")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "5_Heatmap_Correlacion_Genes.png"), dpi=300)
plt.show()

# ============================================================
# 🔬 MODELO PREDICTIVO ALTERNATIVO
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------
# SELECCIÓN DE VARIABLES
# -----------------------------------------
# Predictores
predictor_cols_alt = [
    'Gene_E_Housekeeping',
    'Gene_A_Oncogene',
    'Gene_B_Immune',
    'Gene_C_Stromal',
    'Gene_D_Therapy',
    'Pathway_Score_Inflam'
]

# Definición variable objetivo
le = LabelEncoder()
df['Cell_Type_Encoded'] = le.fit_transform(df['Cell_Type'])

X_alt = df[predictor_cols_alt]
y_alt = df['Cell_Type_Encoded']

# -----------------------------------------
# DIVISIÓN DEL CONJUNTO DE DATOS
# -----------------------------------------
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(
    X_alt, y_alt, test_size=0.3, random_state=42
)

# -----------------------------------------
# ENTRENAMIENTO DEL MODELO
# -----------------------------------------
model_alt = LogisticRegression(max_iter=1000)
model_alt.fit(X_train_alt, y_train_alt)

# -----------------------------------------
# PREDICCIÓN Y EVALUACIÓN
# -----------------------------------------
y_pred_alt = model_alt.predict(X_test_alt)
accuracy_alt = accuracy_score(y_test_alt, y_pred_alt)

print('Accuracy del modelo alternativo:', round(accuracy_alt, 4))
print("\n Reporte de clasificación:\n", classification_report(y_test_alt, y_pred_alt))

# -----------------------------------------
# MATRIZ DE CONFUSIÓN
# -----------------------------------------
plt.figure(figsize=(6,5))
cm_alt = confusion_matrix(y_test_alt, y_pred_alt)
sns.heatmap(cm_alt, annot=True, cmap='Reds', fmt='d')
plt.title('Matriz de Confusión - Modelo Alternativo')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "6_Modelo_Predictivo.png"), dpi=300)
plt.show()
