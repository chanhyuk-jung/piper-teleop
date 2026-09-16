/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import { createSystem, Types, Vector3, Quaternion } from "@iwsdk/core";

export class RobotSystem extends createSystem(
  {},
  {
    scale: { type: Types.Float32, default: 1 },
    serverIp: { type: Types.String, default: `ws://${window.location.hostname}:65432` },
  },
) {
  private coupled = false;
  private socket = new WebSocket(this.config.serverIp.value);

  init(): void {
    this.socket.addEventListener("message", ({ data }) => {
      const event = JSON.parse(data);

      if (event === "VIBRATE") {
        const pad = this.input.xr.gamepads.right;

        if (!pad) return;

        pad.gamepad.vibrationActuator.pulse(0.2, 50);
      }
    });
  }

  update(delta: number, time: number): void {
    if (this.socket.readyState !== WebSocket.OPEN) return;

    const pad = this.input.xr.gamepads.right;
    const raySpace = this.input.xr.xrOrigin.raySpaces.right;

    if (!pad || !raySpace) return;

    let msg = {
      type: "",
      delta: delta,
      timestamp: time,
      payload: {},
    };

    const squeezed = pad.getButtonPressed("xr-standard-squeeze");

    if (this.coupled && squeezed) {
      msg["type"] = "move";
    } else {
      this.coupled = false;
      msg["type"] = "stop";
    }

    if (pad.getButtonDown("xr-standard-squeeze")) {
      this.coupled = true;
      msg["type"] = "start";
    }

    if (pad.getButtonDown("a-button")) {
      this.coupled = false;
      msg["type"] = "reset";
    }

    if (this.coupled) {
      const currentPos = new Vector3();
      const currentQuat = new Quaternion();

      raySpace.getWorldPosition(currentPos);
      raySpace.getWorldQuaternion(currentQuat);

      msg["payload"] = {
        position: [currentPos.x, currentPos.y, currentPos.z],
        quaternion: currentQuat,
        gripper: pad.getButtonValue("xr-standard-trigger"),
      };
    }

    this.socket.send(JSON.stringify(msg));
  }
}
