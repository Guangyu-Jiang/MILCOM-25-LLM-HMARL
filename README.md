# MILCOM-25-LLM-HMARL

This repository contains the implementation of a Hierarchical Multi-Agent Reinforcement Learning (HMARL) system for the CAGE4 (Cyber Agent Games and Experimentation) challenge. The system uses a hierarchical approach where low-level skills are trained separately and then coordinated by a high-level controller.

## Overview

The repository implements a hierarchical reinforcement learning approach for multi-agent coordination in cybersecurity scenarios. The system consists of:

1. **Low-level skill agents**: Specialized agents trained to perform specific tasks (e.g., defensive monitoring, threat removal, traffic control, deception)
2. **High-level coordinator**: A meta-controller that selects which skill to deploy in different situations

## Training Skills

Skills are trained using the `train_skills.py` script. Each skill represents a specialized behavior pattern that agents can use to respond to specific cybersecurity scenarios.

### Usage

```bash
python train_skills.py <skill_name> <output_filename> [--hidden HIDDEN] [--embedding EMBEDDING]
```

### Arguments

- `skill_name`: The name of the skill to train (e.g., "skill_1", "skill_2")
- `output_filename`: Base name for saved model files
- `--hidden`: Dimension of middle layer for actor/critic networks (default: 256)
- `--embedding`: Dimension of node representation for actor/critic networks (default: 128)

### Example

```bash
# Train the defensive monitoring skill
python train_skills.py skill_1 skill1_defensive_monitoring

# Train the threat removal skill
python train_skills.py skill_2 skill2_threat_removal_recover

# Train the traffic control skill
python train_skills.py skill_3 skill3_traffic_control

# Train the deception skill
python train_skills.py skill_4 skill4_deception

# Train the independent PPO baseline (no specialized skill)
python train_skills.py None ppo_baseline
```

> **Note**: Setting the skill name to `None` will train agents without any specialized skill focus, providing an independent PPO baseline for comparison with the hierarchical approach.

### Output

The script will:
- Create directories for logs and model weights if they don't exist
- Train all 5 agents on the specified skill
- Save model checkpoints to `skills/<skill_name>/`
- Save training logs to `logs/`

## Training the Coordinator

After training individual skills, you can train a coordinator agent that learns to select the appropriate skill for each situation using the `train_coordinator.py` script.

### Usage

```bash
python train_coordinator.py <output_filename> [--hidden HIDDEN] [--embedding EMBEDDING]
```

### Arguments

- `output_filename`: Base name for saved model files
- `--hidden`: Dimension of middle layer for actor/critic networks (default: 256)
- `--embedding`: Dimension of node representation for actor/critic networks (default: 128)

### Example

```bash
python train_coordinator.py hier_coordinator
```

### Prerequisites

Before training the coordinator, ensure you have trained all the required skills. The coordinator expects skill models to be available at specific paths:

```
skills/skill_1/skill1_defensive_monitoring-{i}_5k.pt
skills/skill_2/skill2_threat_removal_recover-{i}_5k.pt
skills/skill_3/skill3_traffic_control-{i}_5k.pt
skills/skill_4/skill4_deception-{i}_5k.pt
```

Where `{i}` ranges from 0 to 4 (one model per agent).

### Output

The script will:
- Create directories for logs and model weights if they don't exist
- Train the coordinator to select between the 4 skills
- Save model checkpoints to `coordinator/`
- Save training logs to `logs/`

## System Requirements

- Python 3.7+
- PyTorch
- CybORG cybersecurity simulation environment
- Additional dependencies in `requirements.txt`

## File Structure

- `train_skills.py`: Script for training individual skills
- `train_coordinator.py`: Script for training the skill coordinator
- `models/`: Neural network model definitions
- `wrapper/`: Environment wrappers for interfacing with CybORG
- `skills/`: Directory where trained skill models are saved
- `coordinator/`: Directory where trained coordinator models are saved
- `logs/`: Training logs and metrics # MILCOM-25-LLM-HMARL
