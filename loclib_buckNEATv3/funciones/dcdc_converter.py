#
# Implementación de la evaluación del fitness para el convertidor dcdc
#
import math as m
import numpy as np
import matplotlib.pyplot as plt
import neat
import warnings


class BuckClass:
    """
    Contiene los métodos para:
    - Obtener el método a usar para evaluación del fitness (en función del modelo de
     Buck utilizado y del uso o no de multiprocessing
    - Obtener el método a usar para graficar la salida de la red
    - Métodos de evaluación del fitness para las funciones a aprender
    - Métodos para graficar la salida producida por la ANN (y la función objetivo)
    """

    #steps = 10000
    #steady = 2000

    def __init__(self, dcdc_config):

        #steps = 10000
        #steady = 2000

        self.modelo = dcdc_config['modelo']

        # Componentes del convertidor
        self.l_in = dcdc_config['l_in']
        self.l_out = dcdc_config['l_out']
        self.c_in = dcdc_config['c_in']
        self.c_out = dcdc_config['c_out']
        self.res_hs = dcdc_config['res_hs']
        self.res_ls = dcdc_config['res_ls']

        # Condiciones de operación y evaluación
        self.T = dcdc_config['period']
        self.target_vout = dcdc_config['target_vout']
        self.penalty = dcdc_config['penalty']
        self.tolerancia = dcdc_config['tolerancia']

        # Estado inicial
        self.i_lout = dcdc_config['i_lout_0']
        self.v_out = dcdc_config['v_out_0']
        self.v_ix = dcdc_config['v_ix_0']
        self.i_li = dcdc_config['i_li_0']
        self.duty = dcdc_config['duty_0']

        # Definición test_sequence
        self.type_vin = dcdc_config['type_vin']
        self.type_rload = dcdc_config['type_rload']
        self.vals_vin = dcdc_config['vals_vin']
        self.vals_rload = dcdc_config['vals_rload']

        # Tiempo simulado
        self.steps = int(dcdc_config['steps'])
        self.steady = int(dcdc_config['steady'])

        # Coeficientes del controlador PID
        self.kp = dcdc_config['kp']
        self.ki = dcdc_config['ki']
        self.kd = dcdc_config['kd']
        self.error = np.zeros(self.steps + self.steady)

        # todo: borrar si no lo utilizas al final
        '''
        self.x1 = dcdc_config['v_out_0']
        self.x2 = 0
        self.x3 = 0
        self.x4 = 0
        self.x5 = 0
        self.x6 = 0 #
        '''
        # -----------------------------------------------------------------------
        # Atributos de entradas a manejar - decide funciones a utilizar para
        # generar las secuencias de Vin y Rload
        do1 = f"func_sequence_{self.type_vin}"
        func_sequence_vin = getattr(self, do1)

        do2 = f"func_sequence_{self.type_rload}"
        func_sequence_rload = getattr(self, do2)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # Generación de secuencias Vin y Rload
        self.sequence_vin = func_sequence_vin(self.vals_vin)
        self.sequence_rload = func_sequence_rload(self.vals_rload)
        # ------------------------------------------------------------------------

#   -----------------------------------------------------------------------------------
#   Creacción de secuencias de Vin y Rload. Staticmethods llamados desde __init__
#   -----------------------------------------------------------------------------------
    def func_sequence_constante(self, vals):
        steps = self.steps + self.steady
        ampl = vals[0]
        secuencia = ampl * np.ones(steps)
        return secuencia

    def func_sequence_seno(self, vals):
        """
        Unidades de tiempo se asumen en µsegundos.
        Duración de la secuencia: 10000 pasos <==> 10000 µs => 100 Hz
        :param vals:
        :return:
        """
        nominal_val = vals[0]
        ampl = vals[1]
        freq = vals[2]

        secuencia = np.empty(self.steps + self.steady)
        secuencia[:self.steady] = nominal_val
        for i in range(self.steps):
            secuencia[i + self.steady] = nominal_val + ampl * m.sin(2*m.pi*freq*(i/10**6))
        return secuencia

    def func_sequence_barridof(self, vals):
        nominal_val = vals[0]
        ampl = vals[1]
        f_inic = vals[2]
        f_fin = vals[3]

        secuencia = np.empty(self.steps + self.steady)
        secuencia[:self.steady] = nominal_val

        slope = (f_fin - f_inic) / self.steps

        for i in range(self.steps):
            freq = f_inic + slope * i
            secuencia[i + self.steady] = nominal_val + ampl * m.sin(2 * m.pi * freq * (i / 10 ** 6))
        return secuencia

    def func_sequence_full_sweep(self, vals):
        nominal_val = vals[0]
        ampl = vals[1]
        f_inic = vals[2]
        f_fin = vals[3]

        secuencia = np.empty(self.steps + self.steady)
        secuencia[:self.steady] = nominal_val

        for i in range(self.steps):
            freq = f_inic * 10**(np.log10(f_fin/f_inic)*(i/self.steps))
            secuencia[i + self.steady] = nominal_val + ampl * m.sin(2 * m.pi * freq * (i / 10 ** 6))
        return secuencia

#   -----------------------------------------------------------------------------------
#   Métodos de simulación y evaluación del fitness
#   -----------------------------------------------------------------------------------
    def devuelve_metodo_eval(self, mp):
        """
        Devuelve el método a utilizar para la evaluación del fitness en función del modelo
        :param mp: bool multiprocessing
        :return: retorna como objeto el método a utilizar para evaluar fitness
        """
        x = self.modelo
        if mp:
            do = f"eval_genomes_mp_{x}"
            if hasattr(self, do) and callable(getattr(self, do)):
                func = getattr(self, do)
                return func
        else:
            do = f"eval_genomes_single_{x}"
            if hasattr(self, do) and callable(getattr(self, do)):
                func = getattr(self, do)
                return func

    def devuelve_fitness_eval(self):
        """
        Devuelve el método para obtener el fitness de un genoma según el modelo
        :return:
        """
        x = self.modelo
        do = f"fitness_{x}"
        if hasattr(self, do) and callable(getattr(self, do)):
            func = getattr(self, do)
            return func

    # ---------------- Level 3 - lvl3 ---------------------------------  #
    def eval_genomes_mp_buck_lvl3(self, genomes, config):
        net = neat.nn.FeedForwardNetwork.create(genomes, config)
        genomes.fitness = BuckClass.fitness_buck_lvl3(self, net)
        return genomes.fitness

    def eval_genomes_single_buck_lvl3(self, genomes, config):
        # single process
        for genome_id, genome in genomes:
            # net = RecurrentNet.create(genome, config,1)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = BuckClass.fitness_buck_lvl3(self, net)

    def fitness_buck_lvl3(self, net):
        """
        Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
        :param net:
        :return fitness: El fitness se calcula como 1/e**(-error_total). De esta forma cuando
        el error total es cero el fitness es 1 y conforme aumenta el fitness
        disminuye tendiendo a cero cuando el error tiende infinito
        """
        vout = self.run_buck_simulation_lvl3(net)[1]
        error = (vout - self.target_vout)
        error[0:self.steady] = 0
        error = np.absolute(error)
        #error = np.greater(error, self.tolerancia)*(self.penalty-1)*error + error
        error_tot = error.sum() / self.steps
        return np.exp(-error_tot)

    def run_buck_simulation_lvl3(self, net):

        steps = self.steps + self.steady

        i_lout_record = np.zeros(steps)
        vout_record = np.zeros(steps)
        v_ix_record = np.zeros(steps)
        i_li_record = np.zeros(steps)
        duty_record = np.zeros(steps+1)

        vout_minus_1 = np.ones(steps+1)*self.v_out

        # Ejecuta la simulación
        for i in range(steps):
            duty_record[i] = self.duty
            # Aplica la salida de RN y obtiene nuevo estado. En realidad no hace falta
            # pasar self.duty, puedo leerlo dentro de buck_status_update pero por claridad...
            self.buck_status_update_lvl3(self.sequence_vin[i],
                                         self.sequence_rload[i],
                                         self.duty)
            # Activa la RN y obtiene su salida
            self.duty = net.activate([self.sequence_vin[i],
                                      self.sequence_rload[i],
                                      self.v_out,
                                      vout_minus_1[i], # ])[0]
                                      0,  # self.x2,
                                      0,  # self.x3,
                                      0,  # self.x4,
                                      0,  # self.x5,
                                      0])[0]  # self.x6])[0] '''
            self.duty = min(max(self.duty, 0.01), 0.99)
            # Record estado
            i_lout_record[i] = self.i_lout
            vout_record[i] = self.v_out
            v_ix_record[i] = self.v_ix
            i_li_record[i] = self.i_li

            vout_minus_1[i+1] = self.v_out

        return i_lout_record, vout_record, v_ix_record, i_li_record, duty_record

    # buck_status_update equiv. a do_step
    def buck_status_update_lvl3(self, v_in, r_load, duty):
        """
        Actualiza el estado del convertidor en función de su estado anterior y de la variables
        de entrada (Vin, Vout, D) en el intervalo actual.
        Esta versión de la función se basa en la impedancia de carga (r_load). Debido al
        modelado escogido es el parámetro a utilizar para representar un comportamiento real
        ("no forzado") del convertidor. Calcular i_load como dependiente del valor actual
        de v_out no es correcto.
        :param v_in:
        :param r_load:
        :param duty:
        :return:
        """
        i_load = self.v_out/r_load

        # Cálculo de rizados en output inductance para los 2 subintervalos [0,DT] y [DT, T]
        delta_i_L_DT_T = ((self.v_out + self.res_ls*i_load)*(1-duty)*self.T) / self.l_out                      # ea 7
        delta_i_L_DT = 1 / self.l_out * (self.v_ix - self.v_out - i_load*self.res_hs) * duty * self.T        # eq 6

        i_lout_DT = self.i_lout + delta_i_L_DT                                          # eq 5

        # Corrientes medias en el output inductance en subintervalos del ciclo
        i_lout_avg_0_DT = self.i_lout + delta_i_L_DT/2                                  # eq 2
        i_lout_avg_DT_T = i_lout_DT - delta_i_L_DT_T/2                                  # eq 3

        # Coriente media en la bobina de salida durante el ciclo completo
        i_lout_avg = i_lout_avg_0_DT * duty + i_lout_avg_DT_T * (1 - duty)              # eq 4

        # Actualización de variables de estado del filtro de Salida
        # self.i_lout = self.i_lout + self.T/self.l_out * (self.v_ix*duty - self.v_out -
        #                                                 i_load*(duty*res_hs+(1-duty)*res_ls))   # eq 1
        self.i_lout = i_lout_DT - delta_i_L_DT_T
        self.v_out = self.v_out + 1/self.c_out*(i_lout_avg - i_load)*self.T             # eq 8

        # Actualización de variables de estado del filtro de entrada
        i_Li_T = self.i_li + self.T/self.l_in * (v_in - self.v_ix)                      # eq 9
        v_ix_T = self.v_ix + 1/self.c_in *(self.i_li*self.T - self.i_lout*duty*self.T - # eq 10
                                           1/2*delta_i_L_DT*(duty*self.T)**2)

        self.i_li = i_Li_T
        self.v_ix = v_ix_T

    # -------------------------------------------------------------------------------- #
    # -------------------------- Level 1b - lvl1b ------------------------------------ #
    # -------------------------------------------------------------------------------- #

    def eval_genomes_mp_buck_lvl1b(self, genomes, config):
        net = neat.nn.FeedForwardNetwork.create(genomes, config)
        genomes.fitness = BuckClass.fitness_buck_lvl1b(self, net)
        return genomes.fitness

    def eval_genomes_single_buck_lvl1b(self, genomes, config):
        # single process
        for genome_id, genome in genomes:
            # net = RecurrentNet.create(genome, config,1)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = BuckClass.fitness_buck_lvl1b(self, net)

    def fitness_buck_lvl1b(self, net):
        """
        Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
        :param net:
        :return fitness: El fitness se calcula como 1/e**(-error_total). De esta forma cuando
        el error total es cero el fitness es 1 y conforme aumenta el fitness
        disminuye tendiendo a cero cuando el error tiende infinito
        """
        vout = self.run_buck_simulation_lvl1b(net)[1]
        error = (vout - self.target_vout)
        error[0:self.steady] = 0
        error = np.absolute(error)
        #error = np.greater(error, self.tolerancia)*(self.penalty-1)*error + error
        error_tot = error.sum() / self.steps
        return np.exp(-error_tot)

    def run_buck_simulation_lvl1b(self, net):

        steps = self.steps + self.steady

        i_lout_record = np.zeros(steps)
        vout_record = np.zeros(steps)
        v_ix_record = np.zeros(steps)
        i_li_record = np.zeros(steps)
        duty_record = np.zeros(steps+1)

        vout_minus_1 = np.ones(steps + 1) * self.v_out
        # Ejecuta la simulación
        for i in range(steps):
            duty_record[i] = self.duty
            # Aplica la salida de RN y obtiene nuevo estado. En realidad no hace falta
            # pasar self.duty, puedo leerlo dentro de buck_status_update pero por claridad...
            self.buck_status_update_lvl1b(self.sequence_vin[i],
                                          self.sequence_rload[i],
                                          self.duty)
            # Activa la RN y obtiene su salida
            self.duty = net.activate([self.sequence_vin[i],
                                      self.sequence_rload[i],
                                      self.v_out,
                                      vout_minus_1[i], # ])[0]
                                      0,  # self.x2,
                                      0,  # self.x3,
                                      0,  # self.x4,
                                      0,  # self.x5,
                                      0])[0]  # self.x6])[0] '''
            self.duty = min(max(self.duty, 0.01), 0.99)
            # Record estado
            i_lout_record[i] = self.i_lout
            vout_record[i] = self.v_out
            v_ix_record[i] = self.v_ix
            i_li_record[i] = self.i_li

            vout_minus_1[i + 1] = self.v_out

        return i_lout_record, vout_record, v_ix_record, i_li_record, duty_record, vout_minus_1

    # buck_status_update equiv. a do_step
    def buck_status_update_lvl1b(self, v_in, r_load, duty):
        """
        Actualiza el estado del convertidor en función de su estado anterior y de la variables
        de entrada (Vin, Vout, D) en el intervalo actual.
        Esta versión de la función se basa en la impedancia de carga (r_load). Debido al
        modelado escogido es el parámetro a utilizar para representar un comportamiento real
        ("no forzado") del convertidor. Calcular i_load como dependiente del valor actual
        de v_out no es correcto.
        :param v_in:
        :param r_load:
        :param duty:
        :return:
        """
        i_load = self.v_out/r_load

        # Cálculo de rizados en output inductance para los 2 subintervalos [0,DT]
        delta_iL_0_DT = 1.0 / self.l_out * (v_in - self.i_lout*self.res_hs - self.v_out) * duty * self.T  # eq.6

        # Cálculo del pico de corriente
        ilout_DT = self.i_lout + delta_iL_0_DT  # eq.5

        # Rizado en Lout para [DT, T]
        delta_iL_DT_T = 1.0 / self.l_out * (self.v_out + ilout_DT * self.res_ls) * (1 - duty) * self.T  # eq.7

        # Valores medios de la corrient en los 2 subintervalos
        ilout_avg_DT_T = ilout_DT + delta_iL_DT_T/2                                             # eq.3
        ilout_avg_0_DT = self.i_lout + delta_iL_0_DT                                            # eq.2

        # Valor medio de i_lout en el periodo
        ilout_avg_0_T = ilout_avg_0_DT * duty + (1.0 - duty) * ilout_avg_DT_T                   # eq.4

        # Actualiza i_lout
        self.i_lout = self.i_lout + self.T/self.l_out * \
                      (v_in*duty - self.v_out - ilout_avg_0_T*(self.res_hs*duty + self.res_ls*(1.0-duty)))   # eq.1

        # Actualiza v_out
        self.v_out = self.v_out + 1.0/self.c_out * (ilout_avg_0_T - i_load) * self.T            # eq.8

    def plot_respuesta_buck_lvl1b(self, net, tinic=0, tfinal=None, view=False, filename='salida.svg'):
        """ Plots the population's average and best fitness. """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        simul_results = self.run_buck_simulation_lvl1b(net)

        if tfinal is None:
            tfinal = len(simul_results[0])

        simul_lenght = len(simul_results[0])
        t = np.linspace(0, simul_lenght, simul_lenght)

        fig, axs = plt.subplots(4, 2, figsize=(18, 24))
        fig.suptitle('Evolución temporal')

        axs[0, 0].plot(t[tinic:tfinal], self.sequence_vin)
        axs[0, 0].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vin')

        axs[0, 1].plot(t[tinic:tfinal], self.sequence_rload)
        axs[0, 1].set(xlabel='time (µs)', ylabel='Current (A)', title='Rload')

        axs[1, 0].plot(t[tinic:tfinal], simul_results[3][tinic:tfinal])
        axs[1, 0].set(xlabel='time (µs)', ylabel='Current (A)', title='i_li')

        axs[1, 1].plot(t[tinic:tfinal], simul_results[0][tinic:tfinal])
        axs[1, 1].set(xlabel='time (µs)', ylabel='Current (A)', title='i_lout')

        axs[2, 0].plot(t[tinic:tfinal], simul_results[2][tinic:tfinal])
        axs[2, 0].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vix')

        axs[2, 1].plot(t[tinic:tfinal], simul_results[1][tinic:tfinal])
        axs[2, 1].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vout')

        axs[3, 1].plot(t[tinic:tfinal], simul_results[4][tinic:tfinal])
        axs[3, 1].set(xlabel='time (µs)', ylabel='duty', title='PWM duty')

        axs[3, 0].plot(t[tinic:tfinal], simul_results[5][tinic:tfinal])
        axs[3, 0].set(xlabel='time (µs)', ylabel='Voltage(V)', title='Vout[n-1]')

        for i in range(4):
            for j in range(2):
                axs[i, j].grid(True)

        plt.savefig(filename)
        if view:
            plt.show()
        plt.close()

    # --------------------------------------------------------------------------------- #
    # -------------------------- Level 1b - l1_pid ------------------------------------ #
    # --------------------------------------------------------------------------------- #

    def eval_genomes_mp_buck_l1_pid(self, genomes, config):

        net = neat.nn.FeedForwardNetwork.create(genomes, config)
        genomes.fitness = BuckClass.fitness_buck_l1_pid(self, net)
        return genomes.fitness

    def eval_genomes_single_buck_l1_pid(self, genomes, config):
        # single process
        for genome_id, genome in genomes:
            # net = RecurrentNet.create(genome, config,1)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = BuckClass.fitness_buck_l1_pid(self, net)

    def fitness_buck_l1_pid(self, net):
        """
        Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
        :param net:
        :return fitness: El fitness se calcula como 1/e**(-error_total). De esta forma cuando
        el error total es cero el fitness es 1 y conforme aumenta el fitness
        disminuye tendiendo a cero cuando el error tiende infinito
        """
        vout = self.run_buck_simulation_l1_pid(net)[1]
        error = (vout - self.target_vout)
        error[0:self.steady] = 0
        error = np.absolute(error)
        #error = np.greater(error, self.tolerancia)*(self.penalty-1)*error + error
        error_tot = error.sum() / self.steps
        return np.exp(-error_tot)

    def run_buck_simulation_l1_pid(self, net):

        steps = self.steps + self.steady

        i_lout_record = np.zeros(steps)
        vout_record = np.ones(steps) * self.v_out
        v_ix_record = np.zeros(steps)
        i_li_record = np.zeros(steps)
        duty_record = np.zeros(steps+1)

        u_i = 0
        # Ejecuta la simulación
        for i in range(2, steps-1):
            duty_record[i] = self.duty

            # Aplica la salida de RN y obtiene nuevo estado. En realidad no hace falta
            # pasar self.duty, puedo leerlo dentro de buck_status_update pero por claridad...
            self.buck_status_update_l1_pid(self.sequence_vin[i],
                                           self.sequence_rload[i],
                                           self.duty)

            # Activa la RN y obtiene su salida
            output_ann = net.activate([self.sequence_rload[i],
                                      self.v_out,
                                      vout_record[i-1],  # ])[0]
                                      vout_record[i-2]])[0]   # self.x2,

            self.error[i] = self.target_vout - self.v_out
            u_p = self.kp * self.error[i]
            u_i = u_i + self.ki * self.error[i]
            u_i = min(max(u_i, 0), 1)
            u_d = self.kd * (self.error[i] - self.error[i-1])

            self.duty = u_p + u_i + u_d + 0.5 + output_ann

            self.duty = min(max(self.duty, 0.01), 0.99)
            # Record estado
            i_lout_record[i] = self.i_lout
            vout_record[i] = self.v_out
            v_ix_record[i] = self.v_ix
            i_li_record[i] = self.i_li

        return i_lout_record, vout_record, v_ix_record, i_li_record, duty_record

    # buck_status_update equiv. a do_step
    def buck_status_update_l1_pid(self, v_in, r_load, duty):
        """
        Actualiza el estado del convertidor en función de su estado anterior y de la variables
        de entrada (Vin, Vout, D) en el intervalo actual.
        Esta versión de la función se basa en la impedancia de carga (r_load). Debido al
        modelado escogido es el parámetro a utilizar para representar un comportamiento real
        ("no forzado") del convertidor. Calcular i_load como dependiente del valor actual
        de v_out no es correcto.
        :param v_in:
        :param r_load:
        :param duty:
        :return:
        """
        i_load = self.v_out/r_load

        # Cálculo de rizados en output inductance para los 2 subintervalos [0,DT]
        delta_iL_0_DT = 1.0 / self.l_out * (v_in - self.i_lout*self.res_hs - self.v_out) * duty * self.T  # eq.6

        # Cálculo del pico de corriente
        ilout_DT = self.i_lout + delta_iL_0_DT  # eq.5

        # Rizado en Lout para [DT, T]
        delta_iL_DT_T = 1.0 / self.l_out * (self.v_out + ilout_DT * self.res_ls) * (1 - duty) * self.T  # eq.7

        # Valores medios de la corrient en los 2 subintervalos
        ilout_avg_DT_T = ilout_DT + delta_iL_DT_T/2                                             # eq.3
        ilout_avg_0_DT = self.i_lout + delta_iL_0_DT                                            # eq.2

        # Valor medio de i_lout en el periodo
        ilout_avg_0_T = ilout_avg_0_DT * duty + (1.0 - duty) * ilout_avg_DT_T                   # eq.4

        # Actualiza i_lout
        self.i_lout = self.i_lout + self.T/self.l_out * \
                      (v_in*duty - self.v_out - ilout_avg_0_T*(self.res_hs*duty + self.res_ls*(1.0-duty)))   # eq.1

        # Actualiza v_out
        self.v_out = self.v_out + 1.0/self.c_out * (ilout_avg_0_T - i_load) * self.T            # eq.8

    def plot_respuesta_buck_l1_pid(self, net, tinic=0, tfinal=None, view=False, filename='salida.svg'):
        """ Plots the population's average and best fitness. """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        simul_results = self.run_buck_simulation_l1_pid(net)

        if tfinal is None:
            tfinal = len(simul_results[0])

        simul_lenght = len(simul_results[0])
        t = np.linspace(0, simul_lenght, simul_lenght)

        fig, axs = plt.subplots(4, 2, figsize=(18, 24))
        fig.suptitle('Evolución temporal')

        axs[0, 0].plot(t[tinic:tfinal], self.sequence_vin)
        axs[0, 0].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vin')

        axs[0, 1].plot(t[tinic:tfinal], self.sequence_rload)
        axs[0, 1].set(xlabel='time (µs)', ylabel='Current (A)', title='Rload')

        axs[1, 0].plot(t[tinic:tfinal], simul_results[3][tinic:tfinal])
        axs[1, 0].set(xlabel='time (µs)', ylabel='Current (A)', title='i_li')

        axs[1, 1].plot(t[tinic:tfinal], simul_results[0][tinic:tfinal])
        axs[1, 1].set(xlabel='time (µs)', ylabel='Current (A)', title='i_lout')

        axs[2, 0].plot(t[tinic:tfinal], simul_results[2][tinic:tfinal])
        axs[2, 0].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vix')

        axs[2, 1].plot(t[tinic:tfinal], simul_results[1][tinic:tfinal])
        axs[2, 1].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vout')

        axs[3, 1].plot(t[tinic:tfinal], simul_results[4][tinic:tfinal])
        axs[3, 1].set(xlabel='time (µs)', ylabel='duty', title='PWM duty')

        #axs[3, 0].plot(t[tinic:tfinal], simul_results[5][tinic:tfinal])
        #axs[3, 0].set(xlabel='time (µs)', ylabel='Voltage(V)', title='Vout[n-1]')

        for i in range(4):
            for j in range(2):
                axs[i, j].grid(True)

        plt.savefig(filename)
        if view:
            plt.show()
        plt.close()

    # --------------------------------------------------------------------------------- #
    # -------------------- Level 1b - l1_pid_xloss --------------------------------- #
    # --------------------------------------------------------------------------------- #

    def eval_genomes_mp_buck_l1_pid_xloss(self, genomes, config):

        net = neat.nn.FeedForwardNetwork.create(genomes, config)
        genomes.fitness = BuckClass.fitness_buck_l1_pid_xloss(self, net)
        return genomes.fitness

    def eval_genomes_single_buck_l1_pid_xloss(self, genomes, config):
        # single process
        for genome_id, genome in genomes:
            # net = RecurrentNet.create(genome, config,1)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = BuckClass.fitness_buck_l1_pid_xloss(self, net)

    def fitness_buck_l1_pid_xloss(self, net):
        """
        Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
        :param net:
        :return fitness: El fitness se calcula como 1/e**(-error_total). De esta forma cuando
        el error total es cero el fitness es 1 y conforme aumenta el fitness
        disminuye tendiendo a cero cuando el error tiende infinito
        """
        k = np.log(12)/np.log(3)
        vout = self.run_buck_simulation_l1_pid(net)[1]
        error = (vout - self.target_vout)
        error[0:self.steady] = 0
        error = np.absolute(error)
        error = error + np.greater(error, self.tolerancia)*(error**k - error)
        error_tot = error.sum() / self.steps
        return np.exp(-error_tot)

    def plot_respuesta_buck_l1_pid_xloss(self, net, tinic=0, tfinal=None, view=False, filename='salida.svg'):
        """
        Plots the population's average and best fitness.
        Simplemente llamo a plot_respuesta_buck_l1_pid puesto que las gráficas son exactax. las mismas
        """
        BuckClass.plot_respuesta_buck_l1_pid(self, net, tinic=0, tfinal=None, view=False, filename='salida.svg')


#   ###################################################################################
#               Métodos graficado de resultado obtenido con un genoma dado
#   ###################################################################################
    def devuelve_metodo_graf(self):
        x = self.modelo
        do = f"plot_respuesta_{x}"
        if hasattr(self, do) and callable(getattr(self, do)):
            func = getattr(self, do)
            return func

    def plot_respuesta_buck_lvl3(self, net, tinic=0, tfinal=None, view=False, filename='salida.svg'):
        """ Plots the population's average and best fitness. """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        simul_results = self.run_buck_simulation_lvl3(net)

        if tfinal is None:
            tfinal = len(simul_results[0])

        simul_lenght = len(simul_results[0])
        t = np.linspace(0, simul_lenght, simul_lenght)

        fig, axs = plt.subplots(4, 2, figsize=(18, 24))
        fig.suptitle('Evolución temporal')

        axs[0, 0].plot(t[tinic:tfinal], self.sequence_vin)
        axs[0, 0].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vin')

        axs[0, 1].plot(t[tinic:tfinal], self.sequence_rload)
        axs[0, 1].set(xlabel='time (µs)', ylabel='Current (A)', title='Rload')

        axs[1, 0].plot(t[tinic:tfinal], simul_results[3][tinic:tfinal])
        axs[1, 0].set(xlabel='time (µs)', ylabel='Current (A)', title='i_li')

        axs[1, 1].plot(t[tinic:tfinal], simul_results[0][tinic:tfinal])
        axs[1, 1].set(xlabel='time (µs)', ylabel='Current (A)', title='i_lout')

        axs[2, 0].plot(t[tinic:tfinal], simul_results[2][tinic:tfinal])
        axs[2, 0].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vix')

        axs[2, 1].plot(t[tinic:tfinal], simul_results[1][tinic:tfinal])
        axs[2, 1].set(xlabel='time (µs)', ylabel='Voltage (V)', title='Vout')

        axs[3, 1].plot(t[tinic:tfinal], simul_results[4][tinic:tfinal])
        axs[3, 1].set(xlabel='time (µs)', ylabel='duty', title='PWM duty')

        axs[3, 0].plot(t[tinic:tfinal], simul_results[5][tinic:tfinal])
        axs[3, 0].set(xlabel='time (µs)', ylabel='Voltage(V)', title='Vout[n-1]')


        for i in range(4):
            for j in range(2):
                axs[i, j].grid(True)

        plt.savefig(filename)
        if view:
            plt.show()
        plt.close()

