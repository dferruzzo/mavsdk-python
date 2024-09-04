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
from scipy.interpolate import interp1d
from scipy import signal

# TODO: Implementar a função que gera o sinal de controle de atitude de uma frequência apenas.
# TODO: Implementar uma função que determina o modulo e a fase de Bode para esse sinal.

def multiple_tone_signal():
    """
    Generates a multiple tone signal.
    Returns:
    interpolation_function (callable): A function that interpolates the multiple tone signal.
    """
    t0 = 0.0
    tf = 30.0
    f1 = 0.001
    f2 = 0.004
    f3 = 0.3
    f4 = 2.0
    A = 0.2*180.0/np.pi
    sample_rate = 1e-3
    #
    tempo = np.arange(t0, tf, sample_rate) 
    sinal_tone = A*np.sin(2*np.pi*f1*tempo) + A*np.sin(2*np.pi*f2*tempo) + A*np.sin(2*np.pi*f3*tempo) + A*np.sin(2*np.pi*f4*tempo)
    #
    interpolation_function = interp1d(tempo, sinal_tone, kind='linear', fill_value="extrapolate")
    return interpolation_function

def thrust_adjustment(phi, theta, m, g):
    """
    This function adjusts the thrust to keep the drone in the air.
    """
    return m*g/(np.cos(phi*np.pi/180)*np.cos(theta*np.pi/180))

def double_sawtooth_wave():
    """
    Generates a double sawtooth wave function.
    Returns:
    interpolation_function (callable): A function that interpolates the double sawtooth wave.
    """
    t0 = 0.0
    tf = 30.0
    Th = 2.0 # half period
    f = 1/(2*Th)
    A = 0.6*180.0/np.pi
    sample_rate = 1e-3
    #
    tempo = np.arange(t0, tf, sample_rate) 
    sinal_dente = A*signal.sawtooth(2*np.pi*f*tempo, 0.5)
    #
    interpolation_function = interp1d(tempo, sinal_dente, kind='linear', fill_value="extrapolate")
    return interpolation_function

def square_wave():
    """
    Generates a square wave signal.
    Returns:
        interpolation_function: A function that interpolates the square wave signal.
    """
    t0 = 0.0
    tf = 30.0
    Th = 2.0 # half period
    f = 1/(2*Th)
    A = 0.2*180.0/np.pi
    sample_rate = 1e-3
    #
    tempo = np.arange(t0, tf, sample_rate) 
    sinal_quadrado = A*np.sign(np.sin(2*np.pi*f*tempo))
    #
    interpolation_function = interp1d(tempo, sinal_quadrado, kind='linear', fill_value="extrapolate")
    return interpolation_function

def sweep_signal(t, C1, C2, Trec, wmin, wmax, A):
    """
    Generate a swept signal based on the given parameters.
    Parameters:
    t (array-like): Time values for the signal.
    C1 (float): Constant parameter.
    C2 (float): Constant parameter.
    Trec (float): Recovery time constant.
    wmin (float): Minimum frequency.
    wmax (float): Maximum frequency.
    A (float): Amplitude of the signal.
    Returns:
    array-like: The generated swept signal.
    """
    K = C2*(np.exp((C1*t)/Trec)-1)
    w = wmin + K*(wmax-wmin)
    dt=t[1]-t[0]
    theta = np.cumsum(w)*dt # integration of w(t) with respect to t
    delta_sweep = A*np.sin(theta)
    return delta_sweep

def signal_1():
    """
    This function generates an interpolation function for a sweep frequency signal.
    """
    #
    C1 = 4.0 
    C2 = 0.0187 
    Trec = 13.0 
    fmin = 0.01
    wmin = 2*np.pi*fmin
    fmax = 5.0 
    wmax = 2*np.pi*fmax 
    A = 0.2*180/np.pi
    #
    t0 = 0.0
    tf = 15.0
    Ta = 1/(10*fmax)
    #
    tempo = np.arange(t0, tf, Ta) 
    delta_sweep = sweep_signal(tempo, C1, C2, Trec, wmin, wmax, A)
    #
    interpolation_function = interp1d(tempo, delta_sweep, kind='linear', fill_value="extrapolate")
    return interpolation_function

def one_single_tone_signal(A, f0):
    """
    Generates a one-tone signal.
    Parameters:
    - A (float): Amplitude of the signal in rad.
    - f0 (float): Frequency of the signal in hertz.
    Returns:
    - interpolation_function (callable): Interpolation function that represents the one-tone signal.
    This function generates a one-tone signal with the given amplitude and frequency. The signal is represented by an interpolation function that can be used to obtain the signal value at any given time.  
    """
    t0 = 0.0
    tf = 4*(1/f0) # 4 periods
    Ta = 1/(100*f0) # 100 samples per period
    
    tempo = np.arange(t0, tf, Ta) 
    w0 = 2*np.pi*f0 # frequência angular in rad/seg
    Agrad = A*180/np.pi # amplitude em graus
    #
    interpolation_function = interp1d(tempo, Agrad*np.sin(w0*tempo), kind='linear', fill_value="extrapolate")
    return interpolation_function

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


# --- 
    # Selecione aqui a função que será utilizada para gerar o sinal de controle
    # ---
    # O sinal
    A = 0.2 # amplitude em rad
    f0 = 0.01 # frequência em Hz
    delta_tone = one_single_tone_signal(A, f0)
    # ---
    # Frequencia de amostragem para o envio do sinal de controle
    freq_sp = 50.0 # Hz
    t_sp = 1/freq_sp    
    # ---
    # Duranção do sinal de controle
    t = 0.0
    tf = 2*(1/f0) # 2 periods
    # ---
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
    
    print("-- Start sweep frequency signal at 50% thrust in roll")
    while t <= tf:

        roll = float(delta_tone(t))
        pitch = 0.0
        yaw = 0.0 
        
        g = 9.81 # m/s^2
        m = 0.050 # 0.5/9.81 = 0.051 kg
        
        thrust = thrust_adjustment(roll, pitch, m, g)

        await drone.offboard.set_attitude(Attitude(roll, pitch, yaw, thrust))
        t += t_sp

        await asyncio.sleep(t_sp) # 1/freq_sp
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

async def main():
    # Connect to the drone
    drone = System()
    await drone.connect(system_address="udp://:14540")

    # Get the list of parameters
    all_params = await drone.param.get_all_params()

    # Iterate through all int parameters
    for param in all_params.int_params:
        print(f"{param.name}: {param.value}")

    for param in all_params.float_params:
        print(f"{param.name}: {param.value}")

if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
    #asyncio.run(main())

    
