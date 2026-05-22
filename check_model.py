import joblib

# IMPORTA TU CLASE CLR antes de cargar el joblib
from clr_model import CLR  # <-- ajusta el nombre del archivo si no se llama así

m = joblib.load("best_model.joblib")

print("Cargó OK ✅")
print("Tiene feature_names_in_?:", hasattr(m, "feature_names_in_"))
print("feature_names_in_:", getattr(m, "feature_names_in_", None))