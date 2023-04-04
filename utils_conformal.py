import numpy as np


def conformalize_anytime_nn(
    probs, targets, calib_ids, valid_ids, C: int, L: int, conf_type="smx", alpha=0.05
):
    assert conf_type in ["smx", "aps", "rankings"]
    sizes, coverages = [], []
    for l in range(L):
        cal_smx = probs[l, calib_ids, :]
        cal_labels = targets[calib_ids]
        n = len(cal_labels)

        val_smx = probs[l, valid_ids, :]
        valid_labels = targets[valid_ids]
        n_valid = len(valid_labels)

        q_level = np.ceil((n + 1) * (1 - alpha)) / n

        if conf_type == "smx":
            cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
            qhat = np.quantile(cal_scores, q_level, method="higher")
            conformal_sets = val_smx >= (1 - qhat)
        elif conf_type == "aps":
            cal_pi = cal_smx.argsort(1)[:, ::-1]
            cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
            cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
                range(n), cal_labels
            ]
            qhat = np.quantile(cal_scores, q_level, interpolation="higher")
            val_pi = val_smx.argsort(1)[:, ::-1]
            val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
            conformal_sets = np.take_along_axis(
                val_srt <= qhat, val_pi.argsort(axis=1), axis=1
            )
        elif conf_type == "rankings":
            # TODO: check if this is correct and if it makes sense at all
            rankings = cal_smx.argsort(axis=1)
            _targets = cal_labels.numpy().astype(int)
            cal_scores = []
            n_classes = C - 1
            for i in range(len(rankings)):
                cal_scores.append(
                    n_classes - np.where(rankings[i] == _targets[i])[0][0]
                )
            qhat = np.quantile(cal_scores, q_level, method="higher")
            conformal_sets = val_smx.argsort(axis=1)[:, -qhat:]

        print(l + 1, qhat)

        # print(conformal_sets.sum(axis=1).mean(), conformal_sets.sum(axis=1).std(), np.median(conformal_sets.sum(axis=1)))
        if conf_type != "rankings":
            sizes.append(conformal_sets.sum(axis=1).mean())
            coverages.append(
                conformal_sets[np.arange(n_valid), valid_labels].sum() / n_valid
            )
        else:
            sizes.append(conformal_sets.shape[1])
            coverage = 0
            for i in range(len(valid_labels)):
                if int(valid_labels[i]) in conformal_sets[i]:
                    coverage += 1
            coverages.append(coverage / n_valid)

    return sizes, coverages


def conformalize_anytime_nn_raps(
    probs,
    targets,
    calib_ids,
    valid_ids,
    C: int,
    L: int,
    alpha=0.05,
    lam_reg=0.01,
    k_reg=5,
    disallow_zero_sets=False,
    rand=True,
):
    """
    https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb
    """

    reg_vec = np.array(k_reg*[0,] + (C-k_reg)*[lam_reg,])[None,:]

    sizes, coverages = [], []
    for l in range(L):
        cal_smx = probs[l, calib_ids, :]
        cal_labels = targets[calib_ids].cpu().numpy()
        n = len(cal_labels)

        val_smx = probs[l, valid_ids, :]
        valid_labels = targets[valid_ids].cpu().numpy()
        n_valid = len(valid_labels)

        # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
        cal_pi = cal_smx.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == cal_labels[:, None])[1]
        cal_scores = (
            cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]
            - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
        )
        # Get the score quantile
        qhat = np.quantile(
            cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
        )
        # Deploy
        n_val = val_smx.shape[0]
        val_pi = val_smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
        val_srt_reg = val_srt + reg_vec
        indicators = (
            (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val, 1) * val_srt_reg)
            <= qhat
            if rand
            else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        )
        if disallow_zero_sets:
            indicators[:, 0] = True
        conformal_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)

        print(l + 1, qhat)

        sizes.append(conformal_sets.sum(axis=1).mean())
        coverages.append(
            conformal_sets[np.arange(n_valid), valid_labels].sum() / n_valid
        )

    return sizes, coverages
