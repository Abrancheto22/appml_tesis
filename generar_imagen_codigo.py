import matplotlib.pyplot as plt

def generar_imagen_codigo():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    fig.patch.set_facecolor('#1E1E1E')  # Fondo oscuro estilo VS Code
    ax.set_facecolor('#1E1E1E')
    ax.axis('off')

    codigo = [
        ("# 5. Entrenar el Pipeline optimizado (Imputación + Estandarización + Random Forest)", '#6A9955', True),
        ("pipeline = Pipeline([", '#DCDCAA', False),
        ("    ('imputer', SimpleImputer(strategy='median')),", '#9CDCFE', False),
        ("    ('scaler', StandardScaler()),", '#9CDCFE', False),
        ("    ('classifier', RandomForestClassifier(", '#DCDCAA', False),
        ("        n_estimators=200, max_depth=10, min_samples_leaf=8,", '#4EC9B0', False),
        ("        min_samples_split=16, criterion='entropy', class_weight='balanced',", '#4EC9B0', False),
        ("        random_state=42", '#4EC9B0', False),
        ("    ))", '#DCDCAA', False),
        ("])", '#DCDCAA', False),
        ("", '#FFFFFF', False),
        ("# Ajuste exclusivo con datos de entrenamiento (6,000 casos)", '#6A9955', True),
        ("pipeline.fit(X_train, y_train)", '#DCDCAA', False),
        ("print(\"Pipeline de Machine Learning entrenado exitosamente sin fuga de datos.\")", '#CE9178', False),
    ]

    y = 0.95
    line_height = 0.068
    for line, color, italic in codigo:
        ax.text(0.03, y, line, color=color, fontsize=10, fontfamily='monospace',
                fontstyle='italic' if italic else 'normal', va='top')
        y -= line_height

    output_img = "/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/PROYECTOS_TITULO/appml_tesis/codigo_entrenamiento_pipeline.png"
    plt.savefig(output_img, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0.3)
    plt.close()
    print(f"✅ Imagen de código generada: {output_img}")

if __name__ == '__main__':
    generar_imagen_codigo()
