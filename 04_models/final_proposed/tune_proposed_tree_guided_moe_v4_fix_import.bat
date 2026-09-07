@echo off
setlocal

REM V4 fixes ModuleNotFoundError: cnn_branch_updated
REM It copies dependencies into generated_tuning_runs and patches sys.path.

set SCRIPT_DIR=%~dp0
set PROCESSED_DIR=D:\project\data\merge_paper_ready_tree_cost

echo Tuning proposed Tree-guided MoE model V4 fix import...
echo Script dir: %SCRIPT_DIR%
echo Processed dir: %PROCESSED_DIR%

python "%SCRIPT_DIR%tune_proposed_tree_guided_moe_v4_fix_import.py" ^
  --scripts-dir "%SCRIPT_DIR%" ^
  --processed-dir "%PROCESSED_DIR%"

pause
endlocal
