import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as tdist
import os

dirname = os.path.dirname(os.path.realpath(__file__))
figure_dir = os.path.join(dirname, 'figures')

scen_names = [
    'pink_noise', 'drums+saw', 'string_quartet', 'two_speakers', 'speech+noise'
]
rev_names = ['anech', 'medrev', 'strongrev']

# condition names in figures
condnames = ['M', 'FOA', 'P1', 'P2', 'P1F', 'P2F', 'Harpex', 'R']

# condition names as in the MUSHRA result files
conds = [
    'Mono', 'FOA', 'Param1', 'Param2', 'FOA-Amb-Param1', 'FOA-Amb-Param2',
    'Harpex', 'reference'
]
# condition names differ slightly in the PEMOQ and BAMQ csv files
conds_pemoq_bamq = [
    'mono', 'foa', 'param1', 'param2', 'foa_amb_param1', 'foa_amb_param2',
    'harpex', 'reference_not_computed']

df_bamq = pd.read_csv(os.path.join(dirname, '../objective_evaluation/bamq.csv'))
df_pemoq = pd.read_csv(os.path.join(dirname, '../objective_evaluation/pemoq.csv'))
df_ratings = pd.read_csv(os.path.join(dirname, 'webmushra_results.csv'))
df_ratings['subject'] = np.arange(df_ratings.shape[0]) // 104 # 104 rows per subject

plt.rc('font', size=8)
bamq_color = 'k'
pemoq_color = 'k'
rev_colors = ['#42a4f5', '#9c42f5', '#f542b3']

# save mean ratings and metrics for condition to compute R squared
m_trial_rats_all = []
m_pemoq_all = []
m_bamq_all = []
m_reverb_all = []

for scen_ind, scenario in enumerate(scen_names):
    plt.figure(figsize=(9, 2))

    handles = []
    reverbs = []


    for rev_ind, reverb in enumerate(rev_names):
        TRIAL = scenario + '_' + reverb
        t = TRIAL.replace('+', ' ')
        subdf = df_ratings[df_ratings['trial_id'] == t]
        if len(subdf) > 0:
            ratings_trial = []
            bamq_trial = []
            pemoq_trial = []
            df_bamq_trial = df_bamq[(df_bamq['reverberation'] == reverb) & (df_bamq['scenario'] == scenario)]
            df_pemoq_trial = df_pemoq[(df_pemoq['reverberation'] == reverb) & (df_pemoq['scenario'] == scenario)]
            for cond, cond_pemoq_bamq in zip(conds, conds_pemoq_bamq):
                rat = subdf[subdf['rating_stimulus'] == cond]['rating_score']
                if cond_pemoq_bamq != 'reference_not_computed':
                    pemoq = df_pemoq_trial['score'][df_pemoq_trial['method'] == cond_pemoq_bamq]
                    bamq = df_bamq_trial['score'][df_bamq_trial['method'] == cond_pemoq_bamq]
                    assert len(pemoq) == 1 and len(bamq) == 1, 'something went wrong'
                    pemoq_trial.append(pemoq.to_numpy()[0])
                    bamq_trial.append(bamq.to_numpy()[0])
                    
                    #assert subject order is preserved
                    assert np.all((subdf[subdf['rating_stimulus'] == cond]
                               )['subject'].to_numpy() == np.arange(22))
                ratings_trial.append(rat)

            ratings_trial = np.stack(ratings_trial, axis=0)
            bamq_trial = np.stack(bamq_trial, axis=0)
            pemoq_trial = np.stack(pemoq_trial, axis=0)
            pemoq_mapped = pemoq_trial * 25 + 100
            
            m_trial_rats = np.mean(ratings_trial[:-1, :], axis=1)
            m_trial_rats_all.append(m_trial_rats)
            m_pemoq_all.append(pemoq_mapped)
            m_bamq_all.append(bamq_trial)
            m_reverb_all += 7 * [reverb]

            for i in range(ratings_trial.shape[0]):
                m = np.median(ratings_trial[i, :])
                me = np.mean(ratings_trial[i, :])
                l = np.quantile(ratings_trial[i, :], 0.25)
                u = np.quantile(ratings_trial[i, :], 0.75)

                if scenario == 'pink_noise':
                    shift = 1
                else:
                    shift = rev_ind

                plt.scatter(i + (shift - 1) / 5 +
                            np.linspace(-0.05, 0.05, ratings_trial.shape[1]),
                            ratings_trial[i, :],
                            13,
                            color=rev_colors[rev_ind],
                            alpha=0.2,
                            linewidth=0)
                h = plt.plot([i + (shift - 1) / 5] * 2, [l, u],
                             '-_',
                             color=rev_colors[rev_ind],
                             markersize=8.5)
                plt.plot(i + (shift - 1) / 5,
                         m,
                         '_',
                         color=rev_colors[rev_ind],
                         markersize=13.5)
                plt.plot(i + (shift - 1) / 5,
                         me,
                         'd',
                         color=rev_colors[rev_ind],
                         markersize=7.5)
                if condnames[i] != 'R':
                    h2 = plt.plot(i + (shift - 1) / 5,
                                  bamq_trial[i],
                                  'o',
                                  color=bamq_color,
                                  markersize=4.5,
                                  markerfacecolor='none',
                                  alpha=0.7)
                    h3 = plt.plot(i + (shift - 1) / 5,
                                  pemoq_mapped[i],
                                  'x',
                                  color=pemoq_color,
                                  markersize=4.5,
                                  alpha=0.7)

            handles.append(h[0])

            reverbs.append(reverb)

            rev_ind += 1

    handles.append(h2[0])
    handles.append(h3[0])
    #ax= plt.gca()

    plt.ylim([-10, 110])
    plt.xlim([-0.5, 7.5])
    plt.ylabel('similarity')
    plt.grid(True)
    plt.xticks(np.arange(len(condnames)), condnames)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, scenario + '.pdf'),
                bbox_inches='tight')

    scen_ind += 1
    if scenario == 'speech+noise':
        plt.figure(figsize=(7, 0.4))
        plt.legend(handles,
                   reverbs + ['BAM-Q', 'PEMO-Q ODG' + r'$\, \cdot 25 + 100$'],
                   ncol=5,
                   columnspacing=1.6)
        ax = plt.gca()
        ax.axis('off')
        plt.savefig(os.path.join(figure_dir, 'legend' + '.pdf'),
                    bbox_inches='tight')

def r_sq(data, model):
    return 1 - np.mean((model - data)**2) / np.mean((data - np.mean(data))**2)

m_bamq_all = np.concatenate(m_bamq_all, axis=0)
m_trial_rats_all = np.concatenate(m_trial_rats_all, axis=0)
m_pemoq_all = np.concatenate(m_pemoq_all, axis=0)
m_reverb_all = np.array(m_reverb_all)

print('anech R squared: PEMOQ: %.2f, BAMQ: %.2f' % (r_sq(m_trial_rats_all[m_reverb_all == 'anech'], 
                                                         m_pemoq_all[m_reverb_all == 'anech']), 
                                                    r_sq(m_trial_rats_all[m_reverb_all == 'anech'], 
                                                         m_bamq_all[m_reverb_all == 'anech'])))
print('medrev R squared: PEMOQ: %.2f, BAMQ: %.2f' % (r_sq(m_trial_rats_all[m_reverb_all == 'medrev'], 
                                                         m_pemoq_all[m_reverb_all == 'medrev']), 
                                                    r_sq(m_trial_rats_all[m_reverb_all == 'medrev'], 
                                                         m_bamq_all[m_reverb_all == 'medrev'])))
print('strongrev R squared: PEMOQ: %.2f, BAMQ: %.2f' % (r_sq(m_trial_rats_all[m_reverb_all == 'strongrev'], 
                                                         m_pemoq_all[m_reverb_all == 'strongrev']), 
                                                    r_sq(m_trial_rats_all[m_reverb_all == 'strongrev'], 
                                                         m_bamq_all[m_reverb_all == 'strongrev'])))

YLIM_ABS = [-65, 10]
YLIM_D = [-1.5, 0.75]

SUBRANGE_COND_INDICES = [2, 3, 4, 5]
plt.rc('font', size=10)
for scen_ind, scenario in enumerate(scen_names):
    plt.figure(figsize=(3.1, 2.9))
    plt.rcParams['xtick.bottom'] = True
    plt.subplot2grid((7, 1), (4, 0), rowspan=3)
    plt.plot([-1, 9], [0, 0], 'k--', alpha=0.5)

    rev_ind = 0
    rev_colors = ['#42a4f5', '#9c42f5', '#f542b3']

    handles = []
    reverbs = []
    for rev_ind, reverb in enumerate(rev_names):
        TRIAL = scenario + '_' + reverb
        t = TRIAL.replace('+', ' ')
        subdf = df_ratings[df_ratings['trial_id'] == t]
        if len(subdf) > 0:
            rat_all = []
            for cond in conds:
                rat = subdf[subdf['rating_stimulus'] == cond]['rating_score']
                rat_all.append(rat)

            ratings_trial = np.stack(rat_all, axis=0)

            ratings_trial = ratings_trial - ratings_trial[np.array(condnames) == 'R', :]
            ratings_trial = ratings_trial / np.std(ratings_trial, axis=1, ddof=1)[:,
                                                                         None]
            for i in SUBRANGE_COND_INDICES:
                m = np.mean(ratings_trial[i, :])
                #s = np.std(trial_rats[i, :])
                l = np.quantile(ratings_trial[i, :], 0.25)  #m - s
                u = np.quantile(ratings_trial[i, :], 0.75)  #m + s

                tppf = tdist.ppf(0.975, ratings_trial.shape[1] - 1)
                er = tppf / np.sqrt(ratings_trial.shape[1])

                signif = m + er < 0

                if scenario == 'pink_noise':
                    shift = 1
                else:
                    shift = rev_ind

                if m >= YLIM_D[0]:
                    h = plt.plot([i + (shift - 1) / 5] * 2, [m - er, m + er],
                                 '-',
                                 color=rev_colors[rev_ind],
                                 markersize=8.5)
                    plt.plot(i + (shift - 1) / 5,
                             m,
                             marker='d',
                             color=rev_colors[rev_ind],
                             markersize=8.5)

                else:
                    plt.arrow(i + (shift - 1) / 5,
                              -1.25,
                              0,
                              -0.15,
                              color=rev_colors[rev_ind],
                              width=0.04)

            handles.append(h[0])

            reverbs.append(reverb)

    plt.ylabel('Cohen\'s $d$')
    plt.grid(True)
    plt.xticks(SUBRANGE_COND_INDICES, condnames[2:6])
    plt.yticks(np.arange(-1.5, 1, 0.5))
    plt.xlim([SUBRANGE_COND_INDICES[0] - 0.5, SUBRANGE_COND_INDICES[-1] + 0.5])
    plt.ylim(YLIM_D)

    plt.rcParams['xtick.bottom'] = False
    plt.subplot2grid((7, 1), (0, 0), rowspan=4)
    #plt.plot([-1, 9], [0,0], 'k-.', alpha=0.2)

    for rev_ind, reverb in enumerate(rev_names):
        TRIAL = scenario + '_' + reverb
        t = TRIAL.replace('+', ' ')
        subdf = df_ratings[df_ratings['trial_id'] == t]
        if len(subdf) > 0:
            rat_all = []

            for cond in conds:
                rat = subdf[subdf['rating_stimulus'] == cond]['rating_score']
                assert np.all((subdf[subdf['rating_stimulus'] == cond]
                               )['subject'].to_numpy() == np.arange(22))
                rat_all.append(rat)

            ratings_trial = np.stack(rat_all, axis=0)

            ratings_trial = ratings_trial - ratings_trial[np.array(condnames) == 'R', :]
            ratings_trial = ratings_trial
            for i in SUBRANGE_COND_INDICES:
                me = np.mean(ratings_trial[i, :])
                m = np.median(ratings_trial[i, :])
                l = np.quantile(ratings_trial[i, :], 0.25)
                u = np.quantile(ratings_trial[i, :], 0.75)

                tppf = tdist.ppf(0.975, ratings_trial.shape[1] - 1)
                er = tppf / np.sqrt(ratings_trial.shape[1])

                signif = m + er < 0

                bamq_color = 'k'
                pemoq_color = 'k'
                if scenario == 'pink_noise':
                    shift = 1
                else:
                    shift = rev_ind

                plt.scatter(i + (shift - 1) / 5 +
                            np.linspace(-0.05, 0.05, ratings_trial.shape[1]),
                            ratings_trial[i, :],
                            15,
                            color=rev_colors[rev_ind],
                            alpha=0.2,
                            linewidth=0)

                if m >= YLIM_ABS[0]:
                    h = plt.plot([i + (shift - 1) / 5] * 2, [l, u],
                                 '-_',
                                 color=rev_colors[rev_ind],
                                 markersize=8.5)
                    plt.plot(i + (shift - 1) / 5,
                             m,
                             '_',
                             color=rev_colors[rev_ind],
                             markersize=11)
                    if signif:
                        plt.plot(i + (shift - 1) / 5,
                                 me,
                                 marker='d',
                                 color=rev_colors[rev_ind],
                                 markersize=7.5)
                    else:
                        plt.plot(i + (shift - 1) / 5,
                                 me,
                                 marker='d',
                                 color=rev_colors[rev_ind],
                                 markersize=7.5)
                else:
                    plt.arrow(i + (shift - 1) / 5,
                              -1.25,
                              0,
                              -0.15,
                              color=rev_colors[rev_ind],
                              width=0.04)

            handles.append(h[0])

            reverbs.append(reverb)

    plt.ylabel('difference')

    plt.xticks(SUBRANGE_COND_INDICES, [''] * 4)
    plt.grid(True)
    plt.xlim([1.5, 5.5])
    plt.ylim(YLIM_ABS)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.gcf().align_ylabels()

    plt.savefig(os.path.join(figure_dir, scenario + '_comp.pdf'),
                bbox_inches='tight')

    if scenario == 'speech+noise':
        plt.figure(figsize=(5, 0.5))
        plt.legend(handles[:3], reverbs[:3], ncols=3)
        ax = plt.gca()
        ax.axis('off')
        plt.savefig(os.path.join(figure_dir, 'legend' + '_comp.pdf'),
                    bbox_inches='tight')

COMPARISON_LABELS_ALL = [['P1F', 'P2F'], ['P1', 'P2'], ['P2', 'P2F']]

plt.rc('font', size=10)

for comparison_labels in COMPARISON_LABELS_ALL:
    plt.figure(figsize=(4.3, 2.9))
    ax_abs = plt.subplot2grid((7, 1), (0, 0), rowspan=4)
    ax_d = plt.subplot2grid((7, 1), (4, 0), rowspan=3)
    ax_d.plot([-1, 9], [0, 0], 'k--', alpha=0.5)

    for scen_ind, scenario in enumerate(scen_names):
        for rev_ind, reverb in enumerate(rev_names):
            TRIAL = scenario + '_' + reverb
            t = TRIAL.replace('+', ' ')
            subdf = df_ratings[df_ratings['trial_id'] == t]
            if len(subdf) > 0:
                rat_all = []
                for cond in conds:
                    rat = subdf[subdf['rating_stimulus'] ==
                                cond]['rating_score']
                    assert np.all((subdf[subdf['rating_stimulus'] == cond]
                                   )['subject'].to_numpy() == np.arange(22))
                    rat_all.append(rat)

                ratings_trial = np.stack(rat_all, axis=0)

                

                comparison = (ratings_trial[np.array(condnames) == comparison_labels[0], :] - \
                    ratings_trial[np.array(condnames) == comparison_labels[1], :])[0, :]

                me = np.mean(comparison)
                m = np.median(comparison)
                l = np.quantile(comparison, 0.25)
                u = np.quantile(comparison, 0.75)

                if scenario == 'pink_noise':
                    shift = 1
                else:
                    shift = rev_ind

                ax_abs.scatter(scen_ind + (shift - 1) / 5 +
                               np.linspace(-0.05, 0.05, comparison.shape[0]),
                               comparison,
                               15,
                               color=rev_colors[rev_ind],
                               alpha=0.2,
                               linewidth=0)
                ax_abs.plot([scen_ind + (shift - 1) / 5] * 2, [l, u],
                            '-_',
                            color=rev_colors[rev_ind],
                            markersize=8.5)
                if m >= YLIM_ABS[0]:
                    ax_abs.plot(scen_ind + (shift - 1) / 5,
                                m,
                                '_',
                                color=rev_colors[rev_ind],
                                markersize=11)
                    ax_abs.plot(scen_ind + (shift - 1) / 5,
                                me,
                                marker='d',
                                color=rev_colors[rev_ind],
                                markersize=7.5)
                else:
                    ax_abs.arrow(scen_ind + (shift - 1) / 5,
                                 -1.25,
                                 0,
                                 -0.15,
                                 color=rev_colors[rev_ind],
                                 width=0.04)

                comparison = comparison / np.std(comparison, ddof=1)
                m = np.mean(comparison)
                l = np.quantile(comparison, 0.25)
                u = np.quantile(comparison, 0.75)

                tppf = tdist.ppf(0.975, comparison.shape[0] - 1)
                er = tppf / np.sqrt(comparison.shape[0])

                signif = m + er < 0

                if scenario == 'pink_noise':
                    shift = 1
                else:
                    shift = rev_ind

                if m >= YLIM_D[0]:
                    h = ax_d.plot([scen_ind + (shift - 1) / 5] * 2,
                                  [m - er, m + er],
                                  '-',
                                  color=rev_colors[rev_ind],
                                  markersize=8.5)
                    ax_d.plot(scen_ind + (shift - 1) / 5,
                              m,
                              marker='d',
                              color=rev_colors[rev_ind],
                              markersize=8.5)

                else:
                    ax_d.arrow(scen_ind + (shift - 1) / 5,
                               -1.25,
                               0,
                               -0.15,
                               color=rev_colors[rev_ind],
                               width=0.04)

    ax_abs.set_ylabel('difference')
    ax_abs.grid(True)
    ax_abs.set_xticks(np.arange(0, 5), [''] * 5)
    ax_abs.set_xlim([-0.5, 4.5])
    ax_abs.set_ylim(YLIM_ABS)

    ax_d.set_ylabel('Cohen\'s $d$')
    ax_d.grid(True)
    ax_d.set_xticks(np.arange(0, 5), [
        'noises', 'drums+\nsaw', 'string_\nquartet', 'two_\nspeakers',
        'speech+\nnoise'
    ])
    ax_d.set_xlim([-0.5, 4.5])
    ax_d.set_ylim(YLIM_D)
    ax_d.set_yticks(np.arange(-1.5, 1, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.gcf().align_ylabels()
    plt.savefig(os.path.join(figure_dir,
                             'comp_%s_%s.pdf' % tuple(comparison_labels)),
                bbox_inches='tight')
