#!/usr/bin/env python3

from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> None:
  model_path = Path(__file__).resolve().parent / "xmls" / "scene.xml"
  model = mujoco.MjModel.from_xml_path(str(model_path))
  data = mujoco.MjData(model)
  mujoco.viewer.launch(model, data)


if __name__ == "__main__":
  main()
