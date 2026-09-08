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
    serverIp: { type: Types.String, default: "localhost:65432" },
  },
) {
  private coupled = false;
  private socket = new WebSocket(`ws://${this.config.serverIp.value}`);
  private delta = 0;

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
    this.delta += delta;
    console.log(`dt: ${this.delta} coupled: ${this.coupled}`);

    const pad = this.input.xr.gamepads.right;
    const raySpace = this.input.xr.xrOrigin.raySpaces.right;

    if (!pad || !raySpace) return;

    let event = "TRACK";

    if (pad.getButtonDown("xr-standard-squeeze") && !this.coupled) {
      this.coupled = true;
      event = "START";
    }

    if (pad.getButtonUp("xr-standard-squeeze")) {
      this.coupled = false;
      const data = {
        event: "STOP",
        timestamp: time,
        payload: {},
      };

      this.socket.send(JSON.stringify(data));
      return;
    }

    if (!this.coupled) return;

    const currentPos = new Vector3();
    const currentQuat = new Quaternion();

    raySpace.getWorldPosition(currentPos);
    raySpace.getWorldQuaternion(currentQuat);

    const gripper = pad.getButtonValue("xr-standard-trigger");

    const data = {
      event: event,
      timestamp: time,
      payload: {
        position: [currentPos.x, currentPos.y, currentPos.z],
        quaternion: currentQuat,
        gripper: gripper,
      },
    };

    this.socket.send(JSON.stringify(data));
    this.delta = 0;
  }
}
