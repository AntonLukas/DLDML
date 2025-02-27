import csv
import matplotlib.pyplot as plt
import numpy as np

from openpyxl import load_workbook


def load_file(filepath: str) -> list:
    data = []
    try:
        if '.csv' in filepath:
            with open(filepath, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    data.append(row)
        elif '.xls' in filepath:
            workbook = load_workbook(filename=filepath, data_only=True)
            sheet = workbook.active
            data = list(sheet.values)
            workbook.close()
    except FileNotFoundError as e:
        print(f"File not found. {str(e)}")
        data.append("Error: File not found.")
    except Exception as e:
        print(f"File load failed. {str(e)}")
        data.append("Error: File load failed.")
    parsed_data = data_parser(data)
    return parsed_data


def data_parser(data: list) -> list:
    time_steps = []
    drawdown = []
    try:
        for i in range(len(data)):
            if i != 0:
                time_steps.append(float(data[i][0]))
                drawdown.append(float(data[i][1]))
        output = [time_steps, drawdown]
    except TypeError as e:
        print(f"Non-numerical data encountered. {str(e)}")
        output = ["Error: Non-numerical data encountered."]
    except Exception as e:
        print(f"Data parsing failed. {str(e)}")
        output = ["Error: Data parsing failed."]
    return output


def data_preparation(raw_data: list) -> list:
    smoothed_data = log_distance_smoothing(raw_data)
    derivative_data = calculated_derivative([raw_data[0], smoothed_data[1]])
    return derivative_data


def log_distance_smoothing(data: list) -> list:
    indexes = data[0]
    values = data[1]
    data_count = len(indexes)
    results = []
    for i in range(1, data_count - 1):
        d_log_sum_val = np.log10(values[i - 1]) + np.log10(values[i]) + np.log10(values[i + 1])
        d_log_sum_ind = np.log10(indexes[i - 1]) + np.log10(indexes[i]) + np.log10(indexes[i + 1])
        d_sq_log_sum_ind = (np.log10(indexes[i - 1]) ** 2) + (np.log10(indexes[i]) ** 2) + (
                    np.log10(indexes[i + 1]) ** 2)
        d_log_sum_all = (np.log10(values[i - 1]) * np.log10(indexes[i - 1])) + (
                    np.log10(values[i]) * np.log10(indexes[i])) + (np.log10(values[i + 1]) * np.log10(indexes[i + 1]))
        d_exponent_one = (d_log_sum_val - ((3 * d_log_sum_all - d_log_sum_val * d_log_sum_ind) / (
                    3 * d_sq_log_sum_ind - (d_log_sum_ind ** 2))) * d_log_sum_ind) / 3
        d_exponent_two = (3 * d_log_sum_all - d_log_sum_val * d_log_sum_ind) / (
                    3 * d_sq_log_sum_ind - (d_log_sum_ind ** 2))
        results.append((10.0 ** d_exponent_one) * (indexes[i] ** d_exponent_two))
    return [indexes[1:-1], results]


def calculated_derivative(data: list) -> list:
    indexes = data[0]
    values = data[1]
    data_count = len(values)
    results = []
    d_part_a = (values[0] * np.log10(indexes[1])) + (values[1] * np.log10(indexes[2]))
    d_part_b = (values[0] + values[1]) * (np.log10(indexes[0]) + np.log10(indexes[1]) + np.log10(indexes[2]))
    d_part_c = pow(np.log10(indexes[0]), 2) + pow(np.log10(indexes[1]), 2) + pow(np.log10(indexes[2]), 2)
    d_part_d = pow(np.log10(indexes[0]) + np.log10(indexes[1]) + np.log10(indexes[2]), 2)
    d_derivative = (3 * d_part_a - d_part_b) / (3 * d_part_c - d_part_d)
    results.append(d_derivative)
    for i in range(data_count - 2):
        d_part_a = (values[i] * np.log10(indexes[i + 1])) + (values[i + 1] * np.log10(indexes[i + 2])) + (
                    values[i + 2] * np.log10(indexes[i + 3]))
        d_part_b = (values[i] + values[i + 1] + values[i + 2]) * (
                    np.log10(indexes[i + 1]) + np.log10(indexes[i + 2]) + np.log10(indexes[i + 3]))
        d_part_c = pow(np.log10(indexes[i + 1]), 2) + pow(np.log10(indexes[i + 2]), 2) + pow(np.log10(indexes[i + 3]),
                                                                                             2)
        d_part_d = pow(np.log10(indexes[i + 1]) + np.log10(indexes[i + 2]) + np.log10(indexes[i + 3]), 2)
        d_derivative = (3 * d_part_a - d_part_b) / (3 * d_part_c - d_part_d)
        results.append(d_derivative)
    return [data[0][1:-2], results]


def plot_derivative(derivative_data: list, filename: str) -> None:
    # Create figure
    fig, ax = plt.subplots(dpi=150)
    # Plot the data
    ax.plot(derivative_data[0], derivative_data[1])
    # Customize first figure layout
    ax.loglog()
    ax.axis('equal')
    ax.axis('off')
    ax.legend().set_visible(False)
    plt.savefig(filename, transparent=False)
    plt.close(fig)


def write_class_list(class_list: list, filename: str) -> None:
    try:
        with open(filename, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Filename", "Classes"])
            csv_writer.writerow(class_list)
    except Exception as e:
        print(f"Error saving data: {e}")
