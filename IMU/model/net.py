import torch
import pypose as pp
import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # `gravity` in a TRAIN section used to build the integrator with gravity = 0.0
        # and then print `conf.ngravity`, a key no config in this repo defines -- so it
        # crashed with an AttributeError that named nothing relevant.  It never fired
        # only because `gravity` lives in the DATASET sections (where it is 9.81007 and
        # is consumed by the loader), never in `train`.  Moving it one level up would
        # have silently asked for zero-gravity integration.
        #
        # Now the key means what it says: it sets the integrator's gravity magnitude.
        # Absent, pypose's default (9.81007) is used, which is what every existing
        # config gets and equals UAVdataset.GRAVITY.
        if "gravity" in conf.keys():
            g = float(conf.gravity)
            self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov,
                                                         reset=True, gravity=g)
            print("integrator gravity set from conf:", g)
        else:
            self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov, reset=True)
        print("network constructed: ", self.conf.network, "gtrot: ", self.conf.gtrot)

    def _select(self, data, start, end):

        select = {}
        for k in data.keys():
            if data[k] is None:
                select[k] = None
            else:
                select[k] = data[k][:, start:end]
        return select
    
    def integrate(self, init_state, data, cov_state):
        B, F = data["corrected_acc"].shape[:2]
        inte_pos, inte_vel, inte_rot, inte_cov = [], [], [], []
        gt_rot = None
        if self.conf.gtrot:
            gt_rot = data['rot']
        # `posonly` used to be keyed on the KEY EXISTING, so `posonly: False` still
        # threw the gyro correction away.  It also assigned data['gyro'], which under
        # the padding9 collate is `window_size + 9` long against a `window_size` dt --
        # so the path could not run at all under any current config.  Read the VALUE,
        # and take the same trimmed slice the corrected signals use.
        if self.conf.get("posonly", False):
            data['corrected_gyro'] = data['gyro'][:, -data['dt'].shape[1]:, :]

        # ---- CHUNK LENGTH IS NOT THE LOSS CHECKPOINT DENSITY --------------------
        # `sampling` used to set BOTH: how finely the loss scores the trajectory AND
        # how many times pypose is called.  They are unrelated, and coupling them made
        # the second one enormously expensive for no gain.
        #
        # MEASURED (B=8, W=6000, RTX 4070) -- and pos/vel/rot are IDENTICAL across all
        # of these to 5e-6 relative, because each chunk is initialised from the
        # previous chunk's end state, so chunking is a restatement of one integration:
        #
        #     sampling  50  ->  120 pypose calls  3345 ms   cov (B,120,9,9)
        #     sampling 250  ->   24 pypose calls   481 ms   cov (B, 24,9,9)
        #     no chunking   ->    1 pypose call     48 ms   cov (B,     9,9)
        #
        # 3345 ms against 8 ms for the entire network.  The ONLY thing those 120 calls
        # buy is 120 covariance snapshots, for NLL terms weighted 5.5e-3 / 1.4e-3 with
        # the rotation block at 0.
        #
        # `cov_sampling` is the chunk length; absent, it falls back to `sampling`, so
        # every existing config integrates exactly as before.
        chunk = self.conf.get("cov_sampling", None) or self.conf.sampling
        if chunk:
            inte_state = None
            for iter in range(0, F, chunk):
                if (F - iter) < chunk: continue
                start, end = iter, iter + chunk
                selected_data = self._select(data, start, end)
                selected_cov_state = self._select(cov_state, start, end)

                # take the init sate from last frame as the init state of the next frame
                if inte_state is not None:
                    init_state = {
                        "pos": inte_state["pos"][:,-1:,:],
                        "vel": inte_state["vel"][:,-1:,:],
                        "rot": inte_state["rot"][:,-1:,:],
                    }
                    if self.conf.propcov:
                        init_state["Rij"] = inte_state["Rij"]
                        init_state["cov"] = inte_state["cov"]

                if self.conf.gtrot:
                    gt_rot = selected_data['rot']
                
                ## starting point and ending point                
                inte_state = self.integrator(init_state = init_state, dt = selected_data['dt'], gyro = selected_data['corrected_gyro'],
                            acc = selected_data['corrected_acc'], rot = gt_rot, acc_cov = selected_cov_state['acc_cov'], gyro_cov = selected_cov_state['gyro_cov'])
            
                inte_pos.append(inte_state['pos'])
                inte_rot.append(inte_state['rot'])
                inte_vel.append(inte_state['vel'])
                inte_cov.append(inte_state['cov'])
            
            out_state ={
                'pos': torch.cat(inte_pos, dim =1),
                'vel': torch.cat(inte_vel, dim =1),
                'rot': torch.cat(inte_rot, dim =1),
            }
            if self.conf.propcov:
                out_state['cov'] = torch.stack(inte_cov, dim =1)
        else:
            
            out_state = self.integrator(init_state = init_state, dt = data['dt'], gyro = data['corrected_gyro'],
                            acc = data['corrected_acc'], rot = gt_rot, acc_cov = cov_state['acc_cov'], gyro_cov = cov_state['gyro_cov'])
        
        return {**out_state, **cov_state}

    def inference(self, data):
        '''
        Pure inference, generate the network output.
        '''
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)
        correction = self.decoder(feature)
        
        # Correction update
        data['corrected_acc'] = correction[...,:3] + data["acc"]
        data['corrected_gyro'] = correction[...,3:] + data["gyro"]

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = cov[...,:3]; cov_state['gyro_cov'] = cov[...,3:]

        return {**cov_state, 'correction_acc': correction[...,:3], 'correction_gyro': correction[...,3:]}
 
    ## For reference
    def forward(self, data, init_state):
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)
        correction = self.decoder(feature)

        # Correction update
        data['corrected_acc'] = correction[...,:3] + data["acc"]
        data['corrected_gyro'] = correction[...,3:] + data["gyro"]

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = cov[...,:3]; cov_state['gyro_cov'] = cov[...,3:]

        out_state = self.integrate(init_state = init_state, data = data, cov_state = cov_state)
        return {**out_state, 'correction_acc': correction[...,:3], 'correction_gyro': correction[...,3:]}
