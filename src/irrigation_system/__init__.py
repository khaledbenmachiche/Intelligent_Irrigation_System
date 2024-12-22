"""Irrigation System — modular pipelines for the Intelligent Irrigation project.

Modules
-------
preprocessing : Scaling, outlier removal, sequence building.
model_builder : Parameterized Bidirectional-LSTM construction.
evaluation    : Regression metrics and error analysis.
config        : YAML experiment-config loader.
lstm1_swtd    : SWTD (soil water content) prediction pipeline.
lstm2_cwad    : CWAD (crop yield) prediction pipeline.
common        : Seed control, I/O helpers, and the RunArtifacts dataclass.
"""
