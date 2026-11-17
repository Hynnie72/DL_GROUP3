"""Run the experiments from the terminal (no plots) and save the summary tables as CSV.

Examples:
    python main.py                                   # all 23 experiments, 30 epochs each
    python main.py --epochs 2                        # quick evaluation run
    python main.py --groups "1. Số lớp ẩn" "3. Dropout"
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # no plot windows when running as a script

import pandas as pd

from src import (BASELINE, DEVICE, CLASS_NAMES, SEED, SWEEP_PARAMS, Evaluator, ExperimentRunner,
                 FashionMNISTData, configs_to_dataframe, select_experiments, set_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fashion-MNIST MLP experiments")
    parser.add_argument("--epochs", type=int, default=30, help="epochs per experiment (default: 30)")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="only run these groups (the baseline is always run)")
    parser.add_argument("--log-every", type=int, default=5, help="print every N epochs (default: 5)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--save-dir", default="./saved_models")
    parser.add_argument("--results-dir", default="./results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    set_seed(SEED)
    print("Thiết bị huấn luyện:", DEVICE)

    experiments = select_experiments(epochs=args.epochs, groups=args.groups)
    print(f"Số thí nghiệm sẽ chạy: {len(experiments)} (mỗi thí nghiệm {args.epochs} epoch)")
    print(configs_to_dataframe(experiments).to_string(index=False))

    data = FashionMNISTData(root=args.data_dir)
    evaluator = Evaluator(CLASS_NAMES, data.evaluation_images)
    runner = ExperimentRunner(data, evaluator, baseline_name=BASELINE, save_dir=args.save_dir)
    runner.run_all(experiments, log_every=args.log_every)

    os.makedirs(args.results_dir, exist_ok=True)
    summary = runner.summary()
    ranking = runner.summary(sort_by="Evaluation Acc (%)")
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False, encoding="utf-8-sig")
    ranking.to_csv(os.path.join(args.results_dir, "ranking.csv"), index=False, encoding="utf-8-sig")
    for param in SWEEP_PARAMS:
        table = runner.sweep_table(param)
        if len(table) >= 2:
            table.to_csv(os.path.join(args.results_dir, f"sweep_{param}.csv"), index=False, encoding="utf-8-sig")

    print("\n" + ranking[["Thí nghiệm", "Evaluation Acc (%)", "Macro F1-score", "Chênh lệch Train-Evaluation"]]
          .round(4).to_string())
    print()
    runner.print_conclusions()
    print(f"\nĐã lưu bảng kết quả vào {args.results_dir}/ và trọng số vào {args.save_dir}/")


if __name__ == "__main__":
    main()
