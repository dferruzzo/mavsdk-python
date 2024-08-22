#!/usr/bin/env python3
"""""
 Warning: Only try this in simulation!
          The direct attitude interface is a low level interface to be used
          with caution. On real vehicles the thrust values are likely not
          adjusted properly and you need to close the loop using altitude.

 This script demonstrates how to control the attitude of a drone by setting  
 the pitch, roll, yaw and thrust directly (offboard). This is useful if you
 need a very fast reaction to control the drone. For a more user-friendly 
 interface that allows to set position and velocity setpoints, please refer 
 to the offboard position control example.  

"""
import asyncio

from mavsdk import System
from mavsdk.offboard import (Attitude, OffboardError)

import numpy as np

C1 = 4.0
C2 = 0.0187
Trec = 13.0
wmin = 0.4*2*np.pi # rad/s
wmax = 6.0*2*np.pi # rad/s
A = 0.2*180/np.pi
t = np.linspace(0, 15, 1000) 
dt = t[1]-t[0]  
K = C2*(np.exp((C1*t)/Trec)-1)
w = wmin + K*(wmax-wmin)
theta = np.cumsum(w)*dt # integration of w(t) with respect to t
delta_sweep = A*np.sin(theta)

async def run():
    """ Does Offboard control using attitude commands. """

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()
    
    print("-- Taking off")
    await drone.action.takeoff()

    await asyncio.sleep(20) # 20 seconds

    print("-- Setting initial setpoint")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
              {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return
      
    print("-- Go up at 50% thrust")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.5))
    await asyncio.sleep(2) # 2 seconds
   
    # --- 

    print("-- Start sweep frequency signal at 50% thrust in roll")
    for i in range(len(delta_sweep)):
        await drone.offboard.set_attitude(Attitude(delta_sweep[i], 0.0, 0.0, 0.5))
        await asyncio.sleep(dt)
    await asyncio.sleep(1.0)

    # ---

    print("-- Hover at 50% thrust")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.5))
    await asyncio.sleep(2)

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: \
              {error._result.result}")

    print("-- Returning to Launch")
    await drone.action.return_to_launch()

if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
