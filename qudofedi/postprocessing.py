import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
from datetime import datetime

linear_name_list = ['a', 'absorption']
thirdorder_name_list = ['gsb', 'ground state bleaching', 'se', 'stimulated emission', 'esa', 'excited state absorption']
    
def PostProcess(qjob,
                RF_freq = 0,
                damping_rate = 0,
                T2_value = 0,
                pad =1,
                save_figure = False,
                figure_name = None,
                **pltsavefig_kws,
                ):
    '''
    Method to run sequentially the methods of class PostProcessing: __RotatingFrame(), __FourierTransform(), __PlotSpectra().
    '''
    time_signal = qjob.response_function
    delay_time = qjob.delay_time
    FD_type = qjob.FD_type

    RF_time_signal = RotatingFrame(time_signal, delay_time, FD_type, RF_freq, damping_rate, T2_value)

    (omega, freq_spectra) = FourierTransform(RF_time_signal, delay_time, FD_type, RF_freq, T2_value, pad)

    PlotTimeSignal(RF_time_signal, delay_time, FD_type, save_figure, figure_name, **pltsavefig_kws)

    PlotSpectra(freq_spectra, omega, FD_type, RF_freq, save_figure, figure_name, **pltsavefig_kws)

def RotatingFrame(time_signal,
                  delay_time,
                  FD_type,
                  RF_freq = 0,
                  damping_rate = 0,
                  T2_value = 0,
                  ):
    '''Method to differentiate the application of the rotating frame between linear and non-linear time_signals.
    '''
    if FD_type in linear_name_list:
        return __LinearRotatingFrame(time_signal, delay_time, RF_freq, damping_rate)
    elif FD_type in thirdorder_name_list:
        return __2DRotatingFrame(time_signal, delay_time, RF_freq, damping_rate, T2_value)
    
def __LinearRotatingFrame(time_signal,
                          delay_time,
                          RF_freq = 0,
                          damping_rate = 0,
                          ):
    ''' Method that apply the rotating frame to the linear time_signal.
    '''
    T = np.array(delay_time)
    rf_time_signal = time_signal * np.exp(+1.j * RF_freq * T) * np.exp(- damping_rate * T)
    return rf_time_signal

def __2DRotatingFrame(time_signal,
                      delay_time,
                      RF_freq = 0,
                      damping_rate = 0,
                      T2_value = 0,
                      ):
    ''' Method that apply the rotating frame to the non-linear time_signal.
    '''
    T1 = np.array(delay_time[0])
    T2 = np.array(delay_time[1])
    T3 = np.array(delay_time[2])
    if (T2_value >= len(T2)):
        raise ValueError('T2_value exceed length of T2')
    T1, T3 = np.meshgrid(T1, T3, indexing='ij') 
    rf_time_signal = time_signal[:,T2_value,:] * np.exp(-1.j * RF_freq * (T1-T3)) * np.exp(- damping_rate * (T1+T3))
    return rf_time_signal

def FourierTransform(time_signal,
                     delay_time,
                     FD_type,
                     RF_freq = 0,
                     T2_value = 0,
                     pad = 1,
                     ):
    ''' Method to differentiate the application of the Fourier Transform between linear and non-linear time_signals.
    '''
    if FD_type in linear_name_list:
        return __LinearFourierTransform(time_signal, delay_time, RF_freq, pad)
    elif FD_type in thirdorder_name_list:
        return __2DFourierTransform(time_signal, delay_time, RF_freq, T2_value, pad)
    
def __LinearFourierTransform(time_signal,
                             delay_time,
                             RF_freq = 0,
                             pad = 1,
                             ):
    ''' Method that apply the Fourier Transform to the linear time_signal.
    '''
    dt = delay_time[1] - delay_time[0]
    omega = fftshift(2*np.pi*fftfreq(len(delay_time) * pad, dt)) + RF_freq
    freq_spectra = ifftshift(ifft(np.pad(time_signal, ((0, len(delay_time) * (pad-1))),'constant')))
    return omega, freq_spectra

def __2DFourierTransform(time_signal,
                         delay_time,
                         RF_freq = 0,
                         T2_value = 0,
                         pad = 1,
                         ):
    ''' Method that apply the Fourier Transform to the non-linear time_signal.
    '''
    T1 = delay_time[0]
    dt1 = T1[1] - T1[0]
    T2 = delay_time[1]
    T3 = delay_time[2]
    dt3 = T3[1] - T3[0]
    if (T2_value >= len(T2)):
        raise ValueError('T2_value exceed length of T2')
    omega1 = fftshift(2*np.pi*fftfreq(len(T1) * pad, dt1)) + RF_freq
    omega3 = fftshift(2*np.pi*fftfreq(len(T3) * pad, dt3)) + RF_freq
    omega1, omega3 = np.meshgrid(omega1, omega3, indexing='ij')
    omega = [omega1, omega3]
    time_signal_pad = np.pad(time_signal, ((0,len(T1) * (pad-1)), (0,len(T3) * (pad-1))), 'constant')
    freq_spectra = len(time_signal_pad)/len(time_signal)**2 * fftshift(ifft(fftshift(fft(time_signal_pad, axis=0), axes=0), axis=1), axes=1)
    return omega, freq_spectra

def PlotTimeSignal(time_signal,
                   delay_time,
                   FD_type,
                   save_figure = False,
                   figure_name = None,
                   **pltsavefig_kws,
                   ):
    if FD_type in linear_name_list:
        __LinearPlotTimeSignal(time_signal, delay_time, FD_type, save_figure, figure_name, **pltsavefig_kws)
    elif FD_type in thirdorder_name_list:
        __2DPlotTimeSignal(time_signal, delay_time, FD_type, save_figure, figure_name, **pltsavefig_kws)

def __LinearPlotTimeSignal(time_signal,
                           delay_time,
                           FD_type,
                           save_figure = False,
                           figure_name = None,
                           **pltsavefig_kws,
                           ):
    ''' Method that generate the linear spectrum.
    '''
    plt.plot(delay_time, np.real(time_signal))
    plt.plot(delay_time, np.imag(time_signal))
    plt.xlabel(r'$t$')
    plt.ylabel('Response function')
    if save_figure == True:
        if figure_name == None:
            figure_name = FD_type + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig(figure_name, **pltsavefig_kws)
    plt.show()

def __2DPlotTimeSignal(time_signal, 
                       delay_time, 
                       FD_type, 
                       save_figure = False, 
                       figure_name = None, 
                       **pltsavefig_kws,
                       ):
    ''' Method that generate the non-linear spectrum.
    '''
    real_time_signal = np.real(time_signal)
    T1 = delay_time[0]
    T3 = delay_time[2]
    
    vmin = real_time_signal.min()
    vmax = real_time_signal.max()
    v = max(np.abs(vmin), vmax) 
    levels_contourf = np.linspace(-v, v, 299)
    levels_contour = np.linspace(-v, v, 15)
    ticks = np.linspace(-v, v, 9)        

    plt.plot(T1, T3, 'k:')
    plt.contour(T1, T3, real_time_signal, levels=levels_contour, colors='k', linestyles='solid', linewidths=0.5, vmin=-v, vmax=v)
    plt.contourf(T1, T3, real_time_signal, levels=levels_contourf, cmap='RdBu_r', vmin=-v, vmax=v)
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
    ''' Method that generate the spectrum distinguishing between the linear and the non-linear time_signal.
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
