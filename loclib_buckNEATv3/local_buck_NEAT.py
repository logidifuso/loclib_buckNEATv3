import configparser
import gzip
import json
import multiprocessing
import pickle

import neat
import os
import random
import signal
import shutil
import sys
import time

# import funciones
# Local imports - Namespace packages?
# Este módulo se llama desde main.py, éste módulo no contiene ninguna informacion
# de package, por tanto se resuelve como si fuera top-level, es decir desde el
# directorio "top"

import loclib_buckNEATv3.common.visualize as vis
import loclib_buckNEATv3.funciones.dcdc_converter as func
import loclib_buckNEATv3.common.configuracion as conf

# ::Nota: podría eliminar caso como global variable si paso las funciones de evaluación al
#  modulo correspondiente para cada caso (e.g. seno.py, funcioncita.py, etc..). Se repite
#  código pero es más seguro...
global caso


def create_pool_and_config(config_file, checkpoint):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if checkpoint is not None:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = neat.Population(config)

    return p, config


def evaluate_net_fitness(net, config, caso_x):
    """
    Evalua el mejor genoma encontrado durante la ejecución del experimento
    :param net: mejor individuo encontrado
    :param config:
    :param caso_x: Instancia de la clase Funciones. caso_x.caso indica la función objetivo
    :return: True en caso de éxito; False en caso contrario
    """
    eval_func = func.BuckClass.devuelve_fitness_eval(caso_x)
    fitness = eval_func(net)
    print("El fitnes es:", fitness)
    if fitness < config.fitness_threshold:
        return [fitness, False]
    else:
        return [fitness, True]


def run_experiment(path_results, graphs_path, checkpoints_path, config_file, dcdc_config, checkpoint=None,
                   mp=False, num_generaciones=10, checkpoints_interval=20, verbose=True):

    begin = time.time()

    p, config = create_pool_and_config(config_file, checkpoint)

    # Add a stdout reporter to show progress in the terminal.
    # p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # 2 opciones para guardar checkpointers:
    #   a) Grabar checkpoint sólo cuando el fitness del mejor individuo ha mejorado
    #   b) Grabar checkpoint cada cierto num_generaciones especificado (primer atributo de la Función
    #      Checkpointer
    # Comment out la opción descartada

    #    p.add_reporter(neat.CheckpointerBest(filename_prefix="".join((outputs_dir, '/sin_exp-checkpoint-'))))
    p.add_reporter(neat.Checkpointer(checkpoints_interval,
                                     filename_prefix="".join((checkpoints_path, '\\checkpoint-'))))

    # Crea la instancia dcdc_converter que contiene toda la información del Buck a controlar y
    # encuentra el metodo de evaluación en función del modelo elegido (dado por 'caso')
    dcdc_converter = func.BuckClass(dcdc_config)
    metodo_eval = func.BuckClass.devuelve_metodo_eval(dcdc_converter, mp)
    # todo: quitar
    #print(metodo_eval)

    # encuentra la grafica de resultado
    metodo_graf = func.BuckClass.devuelve_metodo_graf(dcdc_converter)

    pe = None
    # try-except para posibilitar interrupciones manuales (desde teclado) de forma correcta y devolver
    # la población y configuración para evaluar la mejor red
    if True: # todo: comment out if True and de-comment try: except: clause
    #try:
        if mp:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), metodo_eval)

            signal.signal(signal.SIGINT, original_sigint_handler)

            best_genome = p.run(pe.evaluate, num_generaciones)
        else:
            best_genome = p.run(metodo_eval, num_generaciones)

        # Comprobación de si el mejor genoma es un hit (en el caso de las aproximación a Funciones
        # no tiene sentido, ya está la información en "stats". Aquí sólo para futuras evaluaciones de
        # tareas de reinforcement learning.
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        # print("\n\nRe-evaluación del mejor individuo")
        dcdc_converter = func.BuckClass(dcdc_config)
        fitness, hit = evaluate_net_fitness(net, config, dcdc_converter)

        if verbose:
            # Muestra info del mejor genoma
            print('\nBest genome:\n{!s}'.format(best_genome))

            if hit:
                print("ÉXITO!!!")
                print("Fitness =", fitness)
            else:
                print("FRACASO!!!")
                print("Fitness =", fitness)

            end = time.time()
            # Tiempo de ejecución de neat
            print(f'Tiempo de ejecución: {end - begin}')

        # Visualiza los resultados del experimento
        node_names = dcdc_config['inputs']
        node_names[0] = 'output_d'
        vis.draw_net(config, best_genome, view=verbose, node_names=node_names, directory=graphs_path, fmt='svg')

        # todo: Poder escoger entre plot_stats_v1 y plot_stats_v2 según un parámetro en config_exp.
        #  Usar p.e. metodo dinamico con una lista
        # vis.plot_stats_v1(stats, ylog=False, view=True, filename=os.path.join(graphs_path, 'avg_fitness.svg'))
        # vis.plot_stats_sine3(stats, ylog=True, view=True, filename=os.path.join(graphs_path, 'avg_fitness.svg'))

        # Recupera la generación de inicio a partir del checkpoint start si este no era 0 para
        # iniciar correctamente el rango en el eje x
        if checkpoint is not None:
            index = checkpoint.find('checkpoint-')
            gen_start = int(checkpoint[index+11:])   # pues longitud de 'checkpoint-' = 11
        else:
            gen_start = 0

        vis.plot_stats_v2(stats, gen_start, ylog=True, view=verbose,
                          filename=os.path.join(graphs_path, 'avg_fitness.svg'))

        vis.plot_species(stats, gen_start, view=verbose,
                         filename=os.path.join(graphs_path, 'speciation.svg'))

        # unc.Funciones.plot_salida(net, view=True, filename=os.path.join(graphs_path, 'salida.svg'))
        metodo_graf(net, view=verbose,
                    filename=os.path.join(graphs_path, 'salida.svg'))

        stats.save_genome_fitness(delimiter=',',
                                  filename=os.path.join(path_results, 'fitness_history.csv'),
                                  with_cross_validation=False)

        stats.save_species_count(delimiter=',',
                                 filename=os.path.join(path_results, 'speciation.csv'))

        stats.save_species_fitness(delimiter=',',
                                   null_value='NA',
                                   filename=os.path.join(path_results, 'species_fitness.csv'))

        # Guarda la instancia del StatisticsReporter actual en 2 archivos:
        # 1. generation_statistics como .json
        path_to_stats = os.path.join(path_results, 'estadisticas.json')
        with open(path_to_stats, 'w') as stats_file:
            estadisticas = stats.generation_statistics
            # estadisticas = [stats.most_fit_genomes, stats.generation_statistics]
            json.dump(estadisticas, stats_file)

        # 2. most_fit_genomes comprimido by cpickle (or pickle). Con el mejor genome (arquitectura)
        # de cada generación. No se puede grabar como .json (or .csv), ..atributo de la clase statistics
        # Grabo una tupla (config, lista de most_fit_genomes)
        path_to_most_fit_genomes = os.path.join(path_results, 'most_fit_genomes.pkl')
        with gzip.open(path_to_most_fit_genomes, 'w', compresslevel=5) as f:
            data = (config, stats.most_fit_genomes)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
    #except:
        print("Stopping the Jobs. ", sys.exc_info())
        if mp:
            pe.pool.terminate()
            pe.pool.join()
            print("pool ok")
        return p, config

    return p, config


def experiment_configuration_parsing(ruta_config_exp):

    config_exp = configparser.ConfigParser()
    config_exp.read(ruta_config_exp)

    # todo: del anterior experimento, a borrar
    #exper_type = config_exp.get('seccion_0', 'experimento')
    config_file_neat = config_exp.get('seccion_0', 'archivo_config_neat')
    cp = config_exp.getint('seccion_0', 'checkpoint_start')
    n_generaciones = config_exp.getint('seccion_0', 'num_generaciones')
    mp = config_exp.getboolean('seccion_0', 'mp')
    seed = config_exp.get('seccion_0', 'seed')
    checkpoints_interval = config_exp.getint('seccion_0', 'checkpoints_interval')
    ruta_checkpoint_start = config_exp.get('seccion_0', 'ruta_checkpoint_start')
    verbose = config_exp.getboolean('seccion_0', 'verbose')

    dcdc_config = {}
    for (key, val) in config_exp.items('dcdc_config'):
        if key == 'modelo' or key == 'type_vin' or key == 'type_rload':
            dcdc_config[key] = val
        elif key == 'vals_vin' or key == 'vals_rload':
            lista = val.split(', ')
            lista2 = []
            for el in lista:
                lista2.append(float(el))
            dcdc_config[key] = lista2
        elif key == 'inputs':
            list_inputs_names = val.split(', ')
            dicc = {}
            i = -1
            for el in list_inputs_names:
                dicc[i] = el
                i -= 1
            dcdc_config[key] = dicc
        else:
            dcdc_config[key] = float(val)

    return [dcdc_config, config_file_neat, cp, n_generaciones, mp, seed,
            checkpoints_interval, ruta_checkpoint_start, verbose]


def start_experiment(ruta_config_exp, ruta_experiment):
    """
    Se encarga de iniciar el experimento, llamando a la función ```run_experiment``` tras obtener
    todos los parámetros a usar proporcionados por los archivos "ruta_config_exp" y
    "ruta_experiment". Determina y crea la carpeta a utilizar para guardar los resultados así
    como los parámetros a utilizar por el algoritmo NEAT.

    :param str ruta_config_exp: ruta al archivo de configuración de la ejecución
    :param str ruta_experiment: ruta a la carpeta raíz donde se creará la carpeta ('resultados_xx')
    con los resultados
    :return: lo retornado por la función ```run_experiment```: population y config. No se utilizan
    en la última versión.
    """
    settings = experiment_configuration_parsing(ruta_config_exp)

    dcdc_config = settings[0]
    config_file_neat = os.path.join(ruta_experiment, settings[1])
    if settings[2] != 0:
        cp = settings[2]
    else:
        cp = None
    n_generaciones = settings[3]
    mp = settings[4]
    seed = settings[5]
    checkpoints_interval = settings[6]
    ruta_cp_start = settings[7]
    verbose = settings[8]

    # Crea carpetas numeradas como resultados_num para guardar los resultados. La numeración permite
    # retomar ejecuciones previas a partir de un checkpoint y grabar los nuevos resultados en un nuevo
    # folder sin perder los anteriores
    i = 0
    while os.path.exists(os.path.join(ruta_experiment, 'ejecucion_%s' % i)):
        i += 1
    path_results = os.path.join(ruta_experiment, 'ejecucion_%s' % i)
    os.makedirs(path_results)

    # paths a carpetas graphs y chekpoints
    graphs_path = os.path.join(path_results, 'graphs')
    checkpoints_path = os.path.join(path_results, 'checkpoints')

    # TODO: Tienen sentido los if para crear las carpetas? Líneas anteriores => Carpetas nuevasb
    # Desactivado clear_output de utils (si existe la carpeta y resultados anteriores no queremos borrarlos!)
    # Limpia los resultados de la ejecución anterior (si los hubiera) o crea la carpeta a usar para guardarlos
    # utils.clear_output(graphs_path)
    # utils.clear_output(checkpoints_path)
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # Fijo semilla para reproducibilidad. Sí está especificada en el config la usa y si no genera una aleatoria
    # que se graba en config_exp de la ejecución para posterior reproducibilidad
    if seed != "None":
        random.seed(int(seed))
    else:
        random.seed()
        seed = random.randint(1, 10000000000)
        random.seed(seed)
    conf.update_setting(ruta_config_exp, 'seccion_0', 'seed_final', str(seed))
    # Graba config_exp para reproducibilidad si se deseara
    shutil.copy(ruta_config_exp, path_results)
    
    if cp is not None:
        ret = run_experiment(path_results, graphs_path, checkpoints_path, config_file_neat, dcdc_config,
                             checkpoint="".join((ruta_cp_start, '/checkpoint-{}'.format(cp))),
                             mp=mp, num_generaciones=n_generaciones,
                             checkpoints_interval=checkpoints_interval, verbose=verbose)
    else:
        ret = run_experiment(path_results, graphs_path, checkpoints_path, config_file_neat, dcdc_config,
                             mp=mp, num_generaciones=n_generaciones,
                             checkpoints_interval=checkpoints_interval, verbose=verbose)


def dummy():
    print('Hola, dummy!')

if __name__ == "__main__":
    pass
