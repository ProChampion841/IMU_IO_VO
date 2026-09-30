import torch

def imu_seq_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data])

    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }

def custom_collate(data):
    dt = torch.stack([d['dt'] for d in data])
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])
    rot = torch.stack([d['rot'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    input_data = {'dt': dt, 'acc': acc, 'gyro': gyro, 'rot': rot,}
    init_state = {'pos': init_pos, 'vel': init_vel, 'rot': init_rot,}

    # The second (MTI / magnetometer) attitude solution.  Guarded rather than
    # assumed because not every dataset class here emits it -- SeqDataset, the
    # inference-side one, does not.  It is NOT padded by padding_collate, exactly
    # like 'rot': both stay at window_size while acc/gyro grow to
    # window_size + pad_len.
    if 'mti_rot' in data[0]:
        input_data['mti_rot'] = torch.stack([d['mti_rot'] for d in data])
    if 'init_mti_rot' in data[0]:
        init_state['mti_rot'] = torch.stack([d['init_mti_rot'] for d in data])

    # The pitot airspeed.  UNLIKE 'rot' it IS padded by padding_collate, because it
    # is an IMU-rate network input concatenated onto acc/gyro before the CNN, not a
    # state-rate quantity.  Still guarded rather than assumed: SeqDataset (the
    # inference-side dataset) does not emit it.
    if 'airspeed' in data[0]:
        input_data['airspeed'] = torch.stack([d['airspeed'] for d in data])

    return  input_data, init_state, {'gt_pos': gt_pos, 'gt_vel': gt_vel, 'gt_rot': gt_rot, }

def padding_collate(data, pad_len = 1, use_gravity = True):
    B = len(data)
    input_data, init_state, label = custom_collate(data)

    if use_gravity:
        iden_acc_vector = torch.tensor([0.,0.,9.81007], dtype=input_data['dt'].dtype).repeat(B,pad_len,1)
    else:
        iden_acc_vector = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)

    pad_acc = init_state['rot'].Inv() * iden_acc_vector
    pad_gyro = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)

    input_data["acc"] = torch.cat([pad_acc, input_data['acc']], dim =1)
    input_data["gyro"] = torch.cat([pad_gyro, input_data['gyro']], dim =1)

    # Airspeed rides with acc/gyro, so it MUST grow by the same pad_len or the
    # concatenated network input is ragged.  The pad REPEATS THE FIRST REAL SAMPLE
    # rather than inventing a value: acc's pad is a clean rotated gravity vector,
    # i.e. "the aircraft sat still at its initial attitude", and the airspeed
    # consistent with sitting at the window's opening condition is the airspeed it
    # actually had there.  Zero-padding instead would put a ~22 m/s step at the
    # boundary -- a landmark the CNN could read the window start off, with no
    # runtime counterpart.  Same argument as pad_rotation in model/attitude.py.
    if 'airspeed' in input_data:
        va = input_data['airspeed']
        input_data['airspeed'] = torch.cat(
            [va[:, :1].expand(-1, pad_len, -1), va], dim=1)

    return  input_data, init_state, label

collate_fcs ={
    "base": custom_collate,
    "padding": padding_collate,
    "padding9": lambda data: padding_collate(data, pad_len = 9),
    "padding1": lambda data: padding_collate(data, pad_len = 1),
    "Gpadding": lambda data: padding_collate(data, pad_len = 9, use_gravity = False),
}