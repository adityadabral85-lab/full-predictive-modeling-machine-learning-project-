from data_loader import prepare_train_test_data
from evaluate_model import (
    evaluate_models,
    generate_report,
    save_confusion_matrix,
    save_roc_curve,
)
from train_model import save_best_model, train_models


def main():
    x_train, x_test, y_train, y_test = prepare_train_test_data()
    trained_models = train_models(x_train, y_train)
    results_df = evaluate_models(trained_models, x_test, y_test)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    model_path = save_best_model(best_model)

    confusion_matrix_path = save_confusion_matrix(best_model, x_test, y_test)
    roc_curve_path = save_roc_curve(best_model, x_test, y_test)
    report_path = generate_report(
        results_df,
        best_model_name,
        model_path,
        [confusion_matrix_path, roc_curve_path],
    )

    print("Predictive modeling project complete.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model: {model_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
