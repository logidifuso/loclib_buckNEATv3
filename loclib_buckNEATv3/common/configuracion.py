import configparser
import os


def create_config(path):
    """
    Crea el archivo de configuración de experimento.
    :param path: Ruta al archivo de configuración del experimento (no al archivo de configuración de NEAT)
    :return:
    """

    config = configparser.ConfigParser()
    config.add_section("seccion_0")
    config.set("seccion_0", "archivo_config_neat", "config.ini")
    config.set("seccion_0", "checkpoint_start", "0")
    config.set("seccion_0", "num_generaciones", "50")
    config.set("seccion_0", "mp", "True")

    with open(path, "w") as configfile:
        config.write(configfile)


def get_config(path):
    """
    Devuelve el objeto de configuración
    :param path: Ruta al archivo de configuración del experimento (no al archivo de configuración de NEAT)
    :return:
    """
    if not os.path.exists(path):
        create_config(path)

    config = configparser.ConfigParser()
    config.read(path)
    return config


def get_setting(path, section, setting):
    """
    Función "getter" de param de configuración (sólo por implementar un mejor encapsulado para el futuro)
    :param path:
    :param section:
    :param setting:
    :return:
    """
    config = get_config(path)
    valor = config.get(section, setting)
    return valor


def update_setting(path, section, setting, value):
    """
    Escribe el valor introducido (value) en el archivo de configuración
    :param path:
    :param section:
    :param setting:
    :param value:
    :return:
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "w") as configfile:
        config.write(configfile)


def delete_setting(path, section, setting):
    """
    Borra el setting especificado
    :param path:
    :param section:
    :param setting:
    :return:
    """
    config = get_config(path)
    config.remove_option(section, setting)
    with open(path, "w") as configfile:
        config.write(configfile)
