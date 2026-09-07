README – Tune Proposed Model V4 Fix Import

Bản này sửa lỗi:

ModuleNotFoundError: No module named 'cnn_branch_updated'

Nguyên nhân:
File generated_tuning_runs\A_current_pr_auc_seed42.py chạy trong thư mục con generated_tuning_runs.
Python chỉ nhìn thư mục đó trước, nên không thấy file:
- cnn_branch_updated.py
- deepfm_branch_updated.py
- fraud_losses.py

Bản V4 sửa bằng 2 cách:
1. Copy dependency files vào generated_tuning_runs.
2. Tự thêm parent folder vào sys.path trong mỗi generated script.

Cách dùng:
Giải nén ZIP, để tất cả file trong cùng một thư mục, ví dụ:
D:\project\model chinh

Chạy:
python tune_proposed_tree_guided_moe_v4_fix_import.py --max-configs 2

Hoặc double click:
tune_proposed_tree_guided_moe_v4_fix_import.bat

Output:
D:\project\data\merge_paper_ready_tree_cost\proposed_tree_guided_moe_tuning

Summary:
D:\project\data\merge_paper_ready_tree_cost\proposed_tree_guided_moe_tuning\_summary

File tổng hợp:
- tuning_results_combined.csv
- paper_ready_tuning_summary.csv
- best_by_metric.json
