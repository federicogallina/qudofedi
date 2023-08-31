import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
from datetime import datetime
from qudofedi import Qjob

linear_name_list = ['a', 'absorption']
thirdorder_name_list = ['gsb', 'ground state bleaching', 'se', 'stimulated emission', 'esa', 'excited state absorption']
    
def PostProcess(qjob: Qjob,
                RF_freq: float = 0,
                damping_rate: float = 0,
                T2_index: int = 0,
                pad: int = 1,
                save_figure: bool = False,
                figure_name: str | None = None,
                **pltsavefig_kws,
                ):
    '''
    Method to run sequentially the methods of class PostProcessing: __RotatingFrame(), __FourierTransform(), __PlotSpectra().
    '''
    response_function = qjob.response_function
    delay_time = qjob.delay_time
    FD_type = qjob.FD_type

    RF_response_function = RotatingFrame(response_function, delay_time, FD_type, RF_freq, damping_rate, T2_index)

    (omega, freq_spectra) = FourierTransform(RF_response_function, delay_time, FD_type, RF_freq, T2_index, pad)

    PlotTimeSignal(RF_response_function, delay_time, FD_type, save_figure, figure_name, **pltsavefig_kws)

    PlotSpectra(freq_spectra, omega, FD_type, RF_freq, save_figure, figure_name, **pltsavefig_kws)

def RotatingFrame(response_function,
                  delay_time,
                  FD_type,
                  RF_freq = 0,
                  damping_rate = 0,
                  T2_index = 0,
                  ):
    '''Method to differentiate the application of the rotating frame between linear and non-linear response_functions.
    '''
    if FD_type in linear_name_list:
        return __LinearRotatingFrame(response_function, delay_time, RF_freq, damping_rate)
    elif FD_type in thirdorder_name_list:
        return __2DRotatingFrame(response_function, delay_time, RF_freq, damping_rate, T2_index)
    
def __LinearRotatingFrame(response_function,
                          delay_time,
                          RF_freq = 0,
                          damping_rate = 0,
                          ):
    ''' Method that apply the rotating frame to the linear response_function.
    '''
    T = np.array(delay_time)
    rf_response_function = response_function * np.exp(+1.j * RF_freq * T) * np.exp(- damping_rate * T)
    return rf_response_function

def __2DRotatingFrame(response_function,
                      delay_time,
                      RF_freq = 0,
                      damping_rate = 0,
                      T2_index = 0,
                      ):
    ''' Method that apply the rotating frame to the non-linear response_function.
    '''
    T1 = np.array(delay_time[0])
    T2 = np.array(delay_time[1])
    T3 = np.array(delay_time[2])
    if (T2_index >= len(T2)):
        raise ValueError('T2_index exceed length of T2')
    T1, T3 = np.meshgrid(T1, T3, indexing='ij') 
    rf_response_function = response_function[:,T2_index,:] * np.exp(-1.j * RF_freq * (T1-T3)) * np.exp(- damping_rate * (T1+T3))
    return rf_response_function

def FourierTransform(response_function,
                     delay_time,
                     FD_type,
                     RF_freq = 0,
                     T2_index = 0,
                     pad = 1,
                     ):
    ''' Method to differentiate the application of the Fourier Transform between linear and non-linear response_functions.
    '''
    if FD_type in linear_name_list:
        return __LinearFourierTransform(response_function, delay_time, RF_freq, pad)
    elif FD_type in thirdorder_name_list:
        return __2DFourierTransform(response_function, delay_time, RF_freq, T2_index, pad)
    
def __LinearFourierTransform(response_function,
                             delay_time,
                             RF_freq = 0,
                             pad = 1,
                             ):
    ''' Method that apply the Fourier Transform to the linear response_function.
    '''
    dt = delay_time[1] - delay_time[0]
    omega = fftshift(2*np.pi*fftfreq(len(delay_time) * pad, dt)) + RF_freq
    freq_spectra = ifftshift(ifft(np.pad(response_function, ((0, len(delay_time) * (pad-1))),'constant')))
    return omega, freq_spectra

def __2DFourierTransform(response_function,
                         delay_time,
                         RF_freq = 0,
                         T2_index = 0,
                         pad = 1,
                         ):
    ''' Method that apply the Fourier Transform to the non-linear response_function.
    '''
    T1 = delay_time[0]
    dt1 = T1[1] - T1[0]
    T2 = delay_time[1]
    T3 = delay_time[2]
    dt3 = T3[1] - T3[0]
    if (T2_index >= len(T2)):
        raise ValueError('T2_index exceed length of T2')
    omega1 = fftshift(2*np.pi*fftfreq(len(T1) * pad, dt1)) + RF_freq
    omega3 = fftshift(2*np.pi*fftfreq(len(T3) * pad, dt3)) + RF_freq
    omega1, omega3 = np.meshgrid(omega1, omega3, indexing='ij')
    omega = [omega1, omega3]
    response_function_pad = np.pad(response_function, ((0,len(T1) * (pad-1)), (0,len(T3) * (pad-1))), 'constant')
    freq_spectra = len(response_function_pad)/len(response_function)**2 * fftshift(ifft(fftshift(fft(response_function_pad, axis=0), axes=0), axis=1), axes=1)
    return omega, freq_spectra

def PlotTimeSignal(response_function,
                   delay_time,
                   FD_type,
                   save_figure = False,
                   figure_name = None,
                   **pltsavefig_kws,
                   ):
    if FD_type in linear_name_list:
        __LinearPlotTimeSignal(response_function, delay_time, FD_type, save_figure, figure_name, **pltsavefig_kws)
    elif FD_type in thirdorder_name_list:
        __2DPlotTimeSignal(response_function, delay_time, FD_type, save_figure, figure_name, **pltsavefig_kws)

def __LinearPlotTimeSignal(response_function,
                           delay_time,
                           FD_type,
                           save_figure = False,
                           figure_name = None,
                           **pltsavefig_kws,
                           ):
    ''' Method that generate the linear spectrum.
    '''
    plt.plot(delay_time, np.real(response_function))
    plt.plot(delay_time, np.imag(response_function))
    plt.xlabel(r'$t$')
    plt.ylabel('Response function')
    if save_figure == True:
        if figure_name == None:
            figure_name = FD_type + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig(figure_name, **pltsavefig_kws)
    plt.show()

def __2DPlotTimeSignal(response_function, 
                       delay_time, 
                       FD_type, 
                       save_figure = False, 
                       figure_name = None, 
                       **pltsavefig_kws,
                       ):
    ''' Method that generate the non-linear spectrum.
    '''
    real_response_function = np.real(response_function)
    T1 = delay_time[0]
    T3 = delay_time[2]
    
    vmin = real_response_function.min()
    vmax = real_response_function.max()
    v = max(np.abs(vmin), vmax) 
    levels_contourf = np.linspace(-v, v, 299)
    levels_contour = np.linspace(-v, v, 15)
    ticks = np.linspace(-v, v, 9)        

    plt.plot(T1, T3, 'k:')
    plt.contour(T1, T3, real_response_function, levels=levels_contour, colors='k', linestyles='solid', linewidths=0.5, vmin=-v, vmax=v)
    plt.contourf(T1, T3, real_response_function, levels=levels_contourf, cmap='RdBu_r', vmin=-v, vmax=v)
    plt.axis('square')
    plt.xlabel(r'$T_{1}$')
    plt.ylabel(r'$T_{3}$')
    plt.colorbar(ticks=ticks)
    if save_figure == True:
        if figure_name == None:
            figure_name = FD_type + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig(figure_name, **pltsavefig_kws)
    plt.show()        
    
def PlotSpectra(freq_spectra, 
                omega, 
                FD_type, 
                RF_freq, 
                save_figure, 
                figure_name,
                **pltsavefig_kws,
                ):
    ''' Method that generate the spectrum distinguishing between the linear and the non-linear response_function.
    '''
    if FD_type in linear_name_list:
        __LinearPlotSpectra(freq_spectra, omega, FD_type, save_figure, figure_name, **pltsavefig_kws)
    elif FD_type in thirdorder_name_list:
        __2DPlotSpectra(freq_spectra, omega, FD_type, RF_freq, save_figure, figure_name, **pltsavefig_kws)
    
def __LinearPlotSpectra(freq_spectra,
                        omega,
                        FD_type,
                        save_figure,
                        figure_name,
                        **pltsavefig_kws,
                        ):
    ''' Method that generate the linear spectrum.
    '''
    plt.plot(omega, np.real(freq_spectra))
    plt.xlabel(r'$\omega$')
    plt.ylabel('Response function')
    if save_figure == True:
        if figure_name == None:
            figure_name = FD_type + 'spectra' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig(figure_name, **pltsavefig_kws)
    plt.show()

def __2DPlotSpectra(freq_spectra,
                    omega,
                    FD_type,
                    RF_freq,
                    save_figure,
                    figure_name,
                    **pltsavefig_kws,
                    ):
    ''' Method that generate the non-linear spectrum.
    '''
    omega1 = omega[0]
    omega3 = omega[1]
    real_freq_spectra = np.real(freq_spectra)
    
    vmin = real_freq_spectra.min()
    vmax = real_freq_spectra.max()
    v = max(np.abs(vmin), vmax) 
    levels_contourf = np.linspace(-v, v, 299)
    levels_contour = np.linspace(-v, v, 15)
    ticks = np.linspace(-v, v, 9)        

    plt.plot(np.diag(omega1), np.diag(omega3), 'k:')
    plt.contour(omega1, omega3, real_freq_spectra, levels=levels_contour, colors='k', linestyles='solid', linewidths=0.5, vmin=-v, vmax=v)
    plt.contourf(omega1, omega3, real_freq_spectra, levels=levels_contourf, cmap='RdBu_r', vmin=-v, vmax=v)
    plt.axis('square')
    plt.xlabel(r'$\omega_{1}$')
    plt.ylabel(r'$\omega_{3}$')
    plt.colorbar(ticks=ticks)
    if save_figure == True:
        if figure_name == None:
            figure_name = FD_type + 'spectra' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig(figure_name, **pltsavefig_kws)
    plt.show()
