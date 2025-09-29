# CARLA Research Project Setup

This project is for autonomous driving research using CARLA simulator.

## Project Structure
- `tests/` - Basic functionality tests and validation scripts
- `algorithms/` - Research algorithms (path planning, perception, etc.)
- `scenarios/` - Custom driving scenarios
- `data/` - Collected simulation data
- `utils/` - Helper functions and utilities
- `config/` - Configuration files

## Development Guidelines
- Use Python 3.8+ with virtual environment
- Follow PEP 8 coding standards
- Include proper error handling for CARLA connections
- Document all research experiments and results
- Use type hints for better code clarity

## CARLA Specific Notes
- Always check CARLA server connection before running scripts
- Handle CARLA client timeouts gracefully
- Clean up spawned actors after experiments
- Use synchronous mode for deterministic experiments