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

def padding_collate(data, pad_len = 1, use_gravity = True, pad_source = "gt"):
    """Prepend `pad_len` synthetic frames so the CNN's first real token is centred.

    `pad_source` chooses what those frames contain.  It exists because the default,
    "gt", puts a GPS-DERIVED quantity into the encoder input on every window:

        "gt"      pad_acc = init_state['rot'].Inv() * [0,0,g], pad_gyro = 0
                  init_state['rot'] is d['init_rot'] (custom_collate), which
                  datasets/dataset.py sets from self.gt_ori -- the SAME GPS-aided
                  attitude array that produces the pos/vel/rot labels.  So each window
                  opens with 9 frames of noise-free, bias-free gravity at the TRUE
                  attitude, plus 9 frames of exactly-zero gyro.  This happens even under
                  `att_input: none` + `att_source: mti`, i.e. on configs whose comments
                  state they are IMU-only and runtime-reproducible.
        "mti"     same construction from init_state['mti_rot'], the runtime-available
                  attitude.  Removes the GPS dependency, keeps the clean-gravity pad.
        "repeat"  repeat the first REAL acc/gyro sample.  Source-free: no attitude of
                  any kind, and no exactly-zero gyro block.

    MEASURED, deterministically -- one trained checkpoint (velnet_v2c_attitude best_model,
    epoch 35), the same 96 validation windows, ONLY the pad swapped, so the 2.55 m
    run-to-run sd does not apply:

        pad_source   mean body-velocity error
        gt              6.35221 m/s   (what the shipped numbers were measured with)
        mti             6.37700 m/s   +0.390%
        repeat          6.38772 m/s   +0.559%

    i.e. 0.56% of the reported accuracy is bought with attitude a deployed aircraft does
    not have.  Small, but real, and it is the reported number that moves, not the model.

    "repeat" is the honest default for new work and is what `padding9_honest` selects.
    "gt" stays the default HERE so that `padding9`, which every historical config and
    every written-up result used, keeps its exact meaning.

    NOTE this is not the whole story for world-frame metrics: `vel_frame_source: gt`
    (model/velocity_net.py) rotates body -> world with GPS-aided attitude at ALL frames,
    which dwarfs 9 pad frames.  Fixing the pad does not by itself make `pos`/`vel`
    deployable numbers; `vel_body` is the output this affects.

    The argument for "repeat" is the one this function ALREADY makes for airspeed a few
    lines below -- "a landmark the CNN could read the window start off, with no runtime
    counterpart".  The acc/gyro pad was the channel that never got it.
    """
    if pad_source not in ("gt", "mti", "repeat"):
        raise ValueError("pad_source must be gt|mti|repeat, got %r" % (pad_source,))

    B = len(data)
    input_data, init_state, label = custom_collate(data)

    if pad_source == "repeat":
        # No invented values at all: the window opens on the condition it actually had.
        pad_acc = input_data['acc'][:, :1].expand(-1, pad_len, -1)
        pad_gyro = input_data['gyro'][:, :1].expand(-1, pad_len, -1)
    else:
        if use_gravity:
            iden_acc_vector = torch.tensor([0.,0.,9.81007], dtype=input_data['dt'].dtype).repeat(B,pad_len,1)
        else:
            iden_acc_vector = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)

        if pad_source == "mti":
            # custom_collate only publishes this when the dataset emits 'init_mti_rot'
            # (datasets/dataset.py:211, :680).  FAIL rather than fall back to
            # init_state['rot']: a silent fallback would put the GPS attitude back into
            # the input on exactly the configs that asked not to have it, and the run
            # would look like it had been fixed.
            if 'mti_rot' not in init_state:
                raise KeyError(
                    "pad_source='mti' needs init_state['mti_rot'], which this dataset "
                    "does not publish (no 'init_mti_rot' key). Use pad_source='repeat', "
                    "which needs no attitude at all.")
            rot = init_state['mti_rot']
        else:
            rot = init_state['rot']
        pad_acc = rot.Inv() * iden_acc_vector
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
    # Source-free pad: no GPS attitude, no zero-gyro landmark.  See padding_collate's
    # docstring for the measured cost (+0.559% body-velocity error on a fixed checkpoint).
    "padding9_honest": lambda data: padding_collate(data, pad_len = 9, pad_source = "repeat"),
    # Runtime-available attitude instead of GPS, keeping the clean-gravity pad (+0.390%).
    "padding9_mti": lambda data: padding_collate(data, pad_len = 9, pad_source = "mti"),
    "padding1": lambda data: padding_collate(data, pad_len = 1),
    "Gpadding": lambda data: padding_collate(data, pad_len = 9, use_gravity = False),
}