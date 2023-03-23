import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as tdist
import os

figure_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'figures')

scen_names = [
    'pink_noise', 'drums+saw', 'string_quartet', 'two_speakers', 'speech+noise'
]
rev_names = ['anech', 'medrev', 'strongrev']
condnames = ['M', 'FOA', 'P1', 'P2', 'P1F', 'P2F', 'Harpex', 'R']
conds = [
    'Mono', 'FOA', 'Param1', 'Param2', 'FOA-Amb-Param1', 'FOA-Amb-Param2',
    'Harpex', 'reference'
]
rev_scen_names = []

for r in rev_names:
    for s in scen_names:
        rev_scen_names.append(r + '_' + s)
dfbamq = pd.read_csv('../objective_evaluation/bamq.csv', delimiter=' ')
dfbamq.index = rev_scen_names
#dfbamq = pd.DataFrame(dfbamq, dfbamq.reindex(index=rev_scen_names)
dfpsmt = pd.read_csv('../objective_evaluation/psmt.csv', delimiter=' ')
dfpsmt.index = rev_scen_names
#dfpsmt.reindex(index=rev_scen_names, copy=False)
df = pd.read_csv('webmushra_results.csv')
df['subject'] = np.arange(df.shape[0]) // 104

plt.rc('font', size=8)
bamq_color = 'k'
psmt_color = 'k'
rev_colors = ['#42a4f5', '#9c42f5', '#f542b3']
#plt.figure()
for scen_ind, scenario in enumerate(scen_names):
    plt.figure(figsize=(9, 2))

    handles = []
    reverbs = []

    for rev_ind, reverb in enumerate(rev_names):
        TRIAL = scenario + '_' + reverb
        t = TRIAL.replace('+', ' ')
        subdf = df[df['trial_id'] == t]
        if len(subdf) > 0:
            rat_all = []
            for cond in conds:
                rat = subdf[subdf['rating_stimulus'] == cond]['rating_score']
                assert np.all((subdf[subdf['rating_stimulus'] == cond]
                               )['subject'].to_numpy() == np.arange(22))
                rat_all.append(rat)

            trial_rats = np.stack(rat_all, axis=0)

            psmt_trial = dfpsmt.loc[reverb + '_' + scenario].to_numpy()
            bamq_trial = dfbamq.loc[reverb + '_' + scenario].to_numpy()

            for i in range(trial_rats.shape[0]):
                m = np.median(trial_rats[i, :])
                me = np.mean(trial_rats[i, :])
                l = np.quantile(trial_rats[i, :], 0.25)
                u = np.quantile(trial_rats[i, :], 0.75)

                if scenario == 'pink_noise':
                    shift = 1
                else:
                    shift = rev_ind

                plt.scatter(i + (shift - 1) / 5 +
                            np.linspace(-0.05, 0.05, trial_rats.shape[1]),
                            trial_rats[i, :],
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
                                  psmt_trial[i],
                                  'x',
                                  color=psmt_color,
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
    plt.ylabel('score')
    plt.grid(True)
    plt.xticks(np.arange(len(condnames)), condnames)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, scenario + '.pdf'),
                bbox_inches='tight')

    scen_ind += 1
    if scenario == 'speech+noise':
        plt.figure(figsize=(7, 0.4))
        plt.legend(handles,
                   reverbs + ['BAM-Q', 'PEMO-Q PSMt' + r'$\, \cdot 100$'],
                   ncol=5,
                   columnspacing=1.6)
        ax = plt.gca()
        ax.axis('off')
        plt.savefig('C:/Users/p3567/Desktop/webMUSHRA/' + 'legend' + '.pdf',
                    bbox_inches='tight')

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
        subdf = df[df['trial_id'] == t]
        if len(subdf) > 0:
            rat_all = []
            for cond in conds:
                rat = subdf[subdf['rating_stimulus'] == cond]['rating_score']
                rat_all.append(rat)

            trial_rats = np.stack(rat_all, axis=0)

            trial_rats = trial_rats - trial_rats[np.array(condnames) == 'R', :]
            trial_rats = trial_rats / np.std(trial_rats, axis=1, ddof=1)[:,
                                                                         None]
            for i in SUBRANGE_COND_INDICES:
                m = np.mean(trial_rats[i, :])
                #s = np.std(trial_rats[i, :])
                l = np.quantile(trial_rats[i, :], 0.25)  #m - s
                u = np.quantile(trial_rats[i, :], 0.75)  #m + s

                tppf = tdist.ppf(0.975, trial_rats.shape[1] - 1)
                er = tppf / np.sqrt(trial_rats.shape[1])

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
        subdf = df[df['trial_id'] == t]
        if len(subdf) > 0:
            rat_all = []

            for cond in conds:
                rat = subdf[subdf['rating_stimulus'] == cond]['rating_score']
                assert np.all((subdf[subdf['rating_stimulus'] == cond]
                               )['subject'].to_numpy() == np.arange(22))
                rat_all.append(rat)

            trial_rats = np.stack(rat_all, axis=0)

            trial_rats = trial_rats - trial_rats[np.array(condnames) == 'R', :]
            trial_rats = trial_rats
            for i in SUBRANGE_COND_INDICES:
                me = np.mean(trial_rats[i, :])
                m = np.median(trial_rats[i, :])
                l = np.quantile(trial_rats[i, :], 0.25)
                u = np.quantile(trial_rats[i, :], 0.75)

                tppf = tdist.ppf(0.975, trial_rats.shape[1] - 1)
                er = tppf / np.sqrt(trial_rats.shape[1])

                signif = m + er < 0

                bamq_color = 'k'
                psmt_color = 'k'
                if scenario == 'pink_noise':
                    shift = 1
                else:
                    shift = rev_ind

                plt.scatter(i + (shift - 1) / 5 +
                            np.linspace(-0.05, 0.05, trial_rats.shape[1]),
                            trial_rats[i, :],
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
            subdf = df[df['trial_id'] == t]
            if len(subdf) > 0:
                rat_all = []
                for cond in conds:
                    rat = subdf[subdf['rating_stimulus'] ==
                                cond]['rating_score']
                    assert np.all((subdf[subdf['rating_stimulus'] == cond]
                                   )['subject'].to_numpy() == np.arange(22))
                    rat_all.append(rat)

                trial_rats = np.stack(rat_all, axis=0)

                psmt_trial = dfpsmt.iloc[scen_ind + rev_ind * 5].to_numpy()
                bamq_trial = dfbamq.iloc[scen_ind + rev_ind * 5].to_numpy()
                comparison = (trial_rats[np.array(condnames) == comparison_labels[0], :] - \
                    trial_rats[np.array(condnames) == comparison_labels[1], :])[0, :]

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
