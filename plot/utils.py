import torch
import numpy as np

def intensity_event_real_extend(process, events, int_std=1, sampleperevent=10):
    params = process.event_params(events)
    params = [p.repeat_interleave(sampleperevent, dim=0) for p in params]
    params = list(zip(*params))

    old_times = events.int_events[0]
    # Generate new times
    new_times = torch.empty(len(old_times) * sampleperevent)
    for idx in range(len(old_times)):
        dt = old_times[idx] / sampleperevent
        if old_times[idx] == 0:
            continue
        new_time = dt + torch.arange(0, old_times[idx], dt)[:sampleperevent]
        new_times[idx*sampleperevent : (idx+1)*sampleperevent] = new_time

    # Generate history
    total_hist = torch.zeros(len(old_times) + 1)
    total_hist[1:] = torch.cumsum(old_times, 0)
    total_hist += events.first_events[0]
    hist_list = []
    for idx in range(len(old_times)):
        for _ in range(sampleperevent):
            hist_list.append(total_hist[:idx+1])

    # Calculate the intensity values
    intensity = []
    for idx in range(len(new_times)):
        int_val = process.intensity_func(new_times[idx] / int_std, [h / int_std for h in hist_list[idx]], params[idx])
        intensity.append(int_val)

    cum_times = new_times
    for idx in range(len(total_hist) - 1):
        cum_times[idx*sampleperevent : (idx+1)*sampleperevent] += total_hist[idx]

    return cum_times, intensity

def intensity_event_extend(process, events, sampleperevent=10):
    params = process.event_params(events)
    params = [p.repeat_interleave(sampleperevent, dim=0) for p in params]
    params = list(zip(*params))

    old_times = events.int_events[0]
    # Generate new times
    new_times = torch.empty(len(old_times) * sampleperevent)
    for idx in range(len(old_times)):
        dt = old_times[idx] / sampleperevent
        if old_times[idx] == 0:
            continue
        new_time = dt + torch.arange(0, old_times[idx], dt)[:sampleperevent]
        new_times[idx*sampleperevent : (idx+1)*sampleperevent] = new_time

    # Generate history
    total_hist = torch.zeros(len(old_times) + 1)
    total_hist[1:] = torch.cumsum(old_times, 0)
    total_hist += events.first_events[0]
    hist_list = []
    for idx in range(len(old_times)):
        for _ in range(sampleperevent):
            hist_list.append(total_hist[:idx+1])

    # Calculate the intensity values
    intensity = []
    for idx in range(len(new_times)):
        int_val = process.intensity_func(new_times[idx], hist_list[idx], params[idx])
        intensity.append(int_val)

    cum_times = new_times
    for idx in range(len(total_hist) - 1):
        cum_times[idx*sampleperevent : (idx+1)*sampleperevent] += total_hist[idx]

    return cum_times, intensity

def total_variation(true_process, fitted_process, events, sampleperevent):
    _, true_intensity = intensity_event_extend(true_process, events, sampleperevent=sampleperevent)
    _, fitted_intensity = intensity_event_extend(fitted_process, events, sampleperevent=sampleperevent)

    interval_size = events.int_events[0]
    true_intensity = torch.Tensor(true_intensity)
    fitted_intensity = torch.Tensor(fitted_intensity)

    diff = torch.abs(true_intensity - fitted_intensity) ** 2

    tv = 0
    for idx in range(len(interval_size)):
        tv += diff[idx*sampleperevent : (idx+1)*sampleperevent].sum() * interval_size[idx] / sampleperevent
    return tv

def generate_x_positions(num_types, num_plots, width, gap):
    diff = gap + width * num_types
    type_pos = np.linspace(
        -num_types / 2 * width, num_types / 2 * width, num_types)

    start = gap + width * num_types / 2

    positions = [[] for _ in range(num_types)]
    centers = []
    for i in range(num_plots):
        centers.append(start + diff * i)
        for j, t_pos in enumerate(type_pos):
            positions[j].append(start + t_pos + diff * i)

    return positions, centers