import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import norm, lognorm, laplace
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import cmasher as cmr
import os
import tikzplotlib
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation
from matplotlib.font_manager import FontProperties

os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],  # or "TeX Gyre Termes"
#     "mathtext.fontset": "cm",
#     "font.size": 16,
#     "axes.titlepad": 30  # Use 'axes.titlepad' instead of 'title.pad'
# })

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # fully LaTeX-native
    "font.size": 16,
    "axes.titlepad": 30
})


def plot_pdf_multiple(data_dict, num_bins=100, chart_type='bar'):
    """
    Plot the Probability Density Function (PDF) of multiple datasets on top of each other.

    Parameters:
    - data_dict: dict, keys are dataset names and values are numpy arrays (matrices).
    - num_bins: int, number of bins for the histogram.
    - chart_type: str, type of chart ('bar' or 'line').
    """
    # Set up the plot
    plt.figure(figsize=(10, 6))

    i = 1
    for name, data in data_dict.items():
        # Flatten the input arrays to 1D
        data_flat = data.flatten()

        # Calculate the histogram
        hist, bins = np.histogram(data_flat, bins=num_bins, density=True)

        # Plot based on the specified chart type
        if chart_type == 'bar':
            # Bar chart
            plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.5, label=name, edgecolor='black')
        elif chart_type == 'line':
            # Line chart
            plt.plot(bins[:-1], hist, label=name, linewidth=2)

        if i == 1:
            i = i+1
            dns = data_flat
    
    kde = st.gaussian_kde(dns, bw_method=0.5)
    plot_lim = [np.min(dns), np.max(dns)]
    data_range = np.linspace(plot_lim[0], plot_lim[1], num_bins)
    plt.plot(data_range, kde(data_range), 'r--', label=f'Normal Dist, bw=0.5', linewidth=2)

    # Add labels and title
    plt.title('Probability Density Function')
    plt.xlabel('Velocity')
    plt.ylabel('Probability Density')
    plt.yscale('log')

    # Add legend
    plt.legend()
    
    # Add grid for better visualization
    plt.grid(True)
    
    # Show the plot
    # plt.show()


def plot_pdf_eps(data_dict, pdf_min=None, pdf_max=None, log_norm_dist=False, 
                 save=True, log_scale=True, x_min=None, x_max=None):
    """
    Plot the Probability Density Function (PDF) of multiple datasets on top of each other,
    and optionally add a log-normal distribution.

    Parameters:
    - data_dict: dict, keys are dataset names and values are numpy arrays (matrices).
    - num_bins: int, number of bins for the histogram.
    - chart_type: str, type of chart ('bar' or 'line').
    - lognorm_params: tuple or None, parameters for the log-normal distribution (shape, loc, scale).
                      If None, log-normal distribution is not plotted.
    """
    i = 0
    num_bins=100
    chart_type='line'
    fontsize = 32
    color = ['black', 'blue']
    plt.figure(figsize=(10, 6))
    for name, data in data_dict.items():

        data_flat = data.flatten()
        hist, bins = np.histogram(data_flat, bins=num_bins, density=True)

        # Plot based on the specified chart type
        if chart_type == 'bar':
            plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.5, label=name, edgecolor='black')
        
        elif chart_type == 'line':
            plt.plot(bins[:-1], hist, label=name, color=color[i], linewidth=2)
        if i == 0 and log_norm_dist:
            name_dns = name
            shape, loc, scale = lognorm.fit(data_flat,method="MM")
            x = np.linspace(min(bins), max(bins), 1000)
            lognorm_pdf = lognorm.pdf(x, s=shape, loc=loc, scale=scale)
            plt.plot(x, lognorm_pdf, 'r-', label=f'{name_dns} log-normal fit', linewidth=2)
        i += 1
    # pdf_min=1e-8, pdf_max=30 for JHS-Disp
    if log_norm_dist: 
        stats_text = rf'{name_dns} Log-normal fit:   shape ($\sigma$): {shape:.3f},   loc: {loc:.3f},   scale ($e^\mu$): {scale:.3f}' + '\n'
        plt.figtext(0.5, -0.25, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    file_name = 'PDF_Disp_1024Cubed'
    if log_scale: plt.yscale('log')
    else: file_name = file_name + '_nonlog'
    if pdf_min and pdf_max: plt.ylim(pdf_min, pdf_max)
    if x_min and x_max: plt.xlim(x_min, x_max)
    ax = plt.gca()

    yticks = ax.get_yticks()
    # ax.set_yticks(yticks)
    yticklabels = [rf"$10^{{{int(np.log10(tick))}}}$" if i % 2 == 0 else "" for i, tick in enumerate(yticks)]
    ax.set_yticklabels(yticklabels)

    # x-axis always plain
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='x')

    xticks = ax.get_xticks()
    # ax.set_xticks(xticks)
    xticklabels = [f"{tick:.0f}" if i % 2 == 1 else "" for i, tick in enumerate(xticks)]
    ax.set_xticklabels(xticklabels)

    # Tick and border styles
    ax.tick_params(width=2.5, length=6, direction='in')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    bold_font = FontProperties(weight='bold', size=fontsize-4)
    ax.set_xticklabels(xticklabels, fontproperties=bold_font)
    ax.set_yticklabels(yticklabels, fontproperties=bold_font)

    ax.legend(fontsize=fontsize-16, loc='best')

    plt.title('PDF of Dissipation', fontsize=fontsize, pad=70)
    plt.xlabel(r'$\varepsilon$', fontsize=fontsize, labelpad=10)
    plt.ylabel('PDF', fontsize=fontsize, labelpad=20)
    plt.legend()
    # plt.grid(True)
    if save: 
        plt.savefig(f'{file_name}_paper.pdf', dpi=300, bbox_inches='tight')
        tikzplotlib.save(f'{file_name}.tex')


def plot_QR_joint_pdf(Q, R, Qw, Q_Comp=None, R_Comp=None, Qw_Comp=None, cutoff=None, 
                      contour_levels=5, percentile_threshold=90, Qlim=10, Rlim=10, 
                      colormesh=True, label_1=None, label_2=None ,save=True, file_name=None):
    """
    Plot the joint PDF of Q and R with a logarithmic color scale and contour lines.

    Parameters:
    - Q: numpy array, second invariant (Q) values.
    - R: numpy array, third invariant (R) values.
    - bins: int, optional, number of bins for the histogram (default: 100).
    - cmap: str, optional, colormap for the plot (default: 'viridis').
    - contour_levels: int, optional, number of contour levels (default: 7).
    - percentile_threshold: float, optional, percentile threshold for contour selection (default: 95).
    - Qlim, Rlim: float, axis limits for Q and R.

    Returns:
    - None
    """
    cmap='jet'
    bins=100
    fontsize = 32
    # Normalize Q and R
    Q_flat = Q.flatten() / Qw**2
    R_flat = R.flatten() / Qw**3

    # Define bin edges
    R_range = np.linspace(-Rlim, Rlim, num=bins)
    Q_range = np.linspace(-Qlim, Qlim, num=bins)

    # Compute the 2D histogram
    H, Q_edges, R_edges = np.histogram2d(Q_flat, R_flat, bins=[Q_range, R_range], density=True)
    print(f'H.min() before = {H.min()}')
    print(f'H.max() before = {H.max()}')
    # H[H == H.min()] = 1.1*H.max()
    # print(f'2nd H.min() before = {H.min()}')
    H_max = 9 #JHS - ForcedIsotropic Dataset
    # H_max = 8 #Daniel - ForcedIsotropic Dataset
    H_min = 1e-8 
    H[H == H.min()] = H_min
    # H[H == 0] = H_min  # Avoid log(0) issues
    # H[H == 0] = H_max*10
    H[H > H_max] = H_max # to make the colorbars the same as DNS
    print(f'H.min() after = {H.min()}')
    print(f'H.max() after = {H.max()}\n')
    # Compute bin centers
    Q_centers = 0.5 * (Q_edges[:-1] + Q_edges[1:])
    R_centers = 0.5 * (R_edges[:-1] + R_edges[1:])

    # Create figure
    plt.figure(figsize=(8, 6))

    # Separatrix curve: Q = -((27/4) * R^2)^(1/3)
    seperatix_R = np.linspace(-Rlim, Rlim, num=1000)
    seperatix_Q = -((27.0 / 4.0) * (seperatix_R**2))**(1.0 / 3.0)
    plt.plot(seperatix_R, seperatix_Q, color='red', alpha=0.5)
    separatrix_legend = mlines.Line2D([], [], color='red', alpha=0.5, label=r'$\frac{27}{4} R^2 + Q^3 = 0$')

    if colormesh:
        # print('H_max = ', H.max())
        # print('H_min = ', H.min())
        plt.pcolormesh(R_centers, Q_centers, H, shading='auto', cmap=cmap, norm=mcolors.LogNorm(vmin=H_min, vmax=H_max))
        plt.colorbar(label='Log Probability Density')
        plt.legend(handles=[separatrix_legend])

    # Logarithmic color scale for pcolormesh
    if contour_levels > 0:
        # Log-spaced contour levels
        contour_min = np.percentile(H, percentile_threshold)
        contour_levels_values = np.logspace(np.log10(contour_min), np.log10(H.max()), contour_levels)
        # Plot contour lines
        plt.contour(R_centers, Q_centers, H, levels=contour_levels_values, colors='black', linewidths=1.5)
        if colormesh:
            if label_1 is not None: contour_legend = mlines.Line2D([], [], color='black', linewidth=1.5, label=label_1)
            else: 
                if cutoff is not None: contour_legend = mlines.Line2D([], [], color='black', linewidth=1.5, label=f'Comp cutoff {cutoff}')
                else: contour_legend = mlines.Line2D([], [], color='black', linewidth=1.5, label='DNS')
        else:
            if label_1 is None: contour_legend = mlines.Line2D([], [], color='black', linewidth=1.5, label='DNS')
            else: contour_legend = mlines.Line2D([], [], color='black', linewidth=1.5, label=label_1)
    if Q_Comp is None:
        # Plot contour lines
        plt.legend(handles=[contour_legend, separatrix_legend])
    else:
        # Normalize Q and R
        Q_comp_flat = Q_Comp.flatten() / Qw_Comp**2
        R_comp_flat = R_Comp.flatten() / Qw_Comp**3

        # Compute the 2D histogram
        H_comp, _, _ = np.histogram2d(Q_comp_flat, R_comp_flat, bins=[Q_range, R_range], density=True)
        # H_comp, Q_edges_comp, R_edges_comp = np.histogram2d(Q_comp_flat, R_comp_flat, bins=[Q_range, R_range], density=True)
        H_comp[H_comp == 0] = H_min  # Avoid log(0) issues

        # Log-spaced contour levels
        contour_min_comp = np.percentile(H_comp, percentile_threshold)
        contour_levels_values_comp = np.logspace(np.log10(contour_min_comp), np.log10(H_comp.max()), contour_levels)

        # Plot contour lines
        plt.contour(R_centers, Q_centers, H_comp, levels=contour_levels_values_comp, colors='blue', linewidths=1.5)
        if label_2 is None: contour_legend_comp = mlines.Line2D([], [], color='blue', linewidth=1.5, label=f'Comp cutoff {cutoff}')
        else: contour_legend_comp = mlines.Line2D([], [], color='blue', linewidth=1.5, label=label_2)
        plt.legend(handles=[contour_legend, contour_legend_comp, separatrix_legend])

    plt.xlabel(r'$R/Q_{w}^{\frac{3}{2}}$')
    plt.ylabel(r'$Q/Q_{w}$')
    plt.xlim([-Rlim, Rlim])
    plt.ylim([-Qlim, Qlim])
    plt.title('Joint PDF of Q and R')
    
    ax = plt.gca()
    # Tick label logic
    yticks = ax.get_yticks()
    # ax.set_yticks(yticks)
    yticklabels = [f"{tick:.0f}" if i % 2 == 1 else "" for i, tick in enumerate(yticks)]
    ax.set_yticklabels(yticklabels)

    # x-axis always plain
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='x')

    xticks = ax.get_xticks()
    # ax.set_xticks(xticks)
    xticklabels = [f"{tick:.0f}" if i % 2 == 0 else "" for i, tick in enumerate(xticks)]
    ax.set_xticklabels(xticklabels)

    # Tick and border styles
    ax.tick_params(width=2.5, length=6, direction='in')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    bold_font = FontProperties(weight='bold', size=fontsize-4)
    ax.set_xticklabels(xticklabels, fontproperties=bold_font)
    ax.set_yticklabels(yticklabels, fontproperties=bold_font)

    # ax.legend(fontsize=fontsize-16, loc='best')
    # plt.grid(True)
    plt.tight_layout()
    if save: 
        plt.savefig(f'{file_name}.pdf', dpi=300, bbox_inches='tight')
    
    if colormesh is False:
        ax = plt.gca()
        collections = ax.collections
        # Check if collections are not empty
        if all(len(coll.get_paths()) == 0 for coll in collections):
            print(f"[Warning] Empty path collection detected. Skipping TikZ export: {file_name}.tex")
        else:
            tikzplotlib.save(f'{file_name}.tex')
        # if colormesh:
        #     if cutoff is None: file_name = 'QR_Joint'
        #     else: file_name = f'PDF_QR_Joint_cutoff_{cutoff}'
        #     plt.savefig(f'PDF_{file_name}.png', dpi=300, bbox_inches='tight')
        # else:
        #     if cutoff is None: file_name = 'QR_Joint_contourlines'
        #     else: file_name = f'QR_Joint_cutoff_{cutoff}_contourlines'
        #     plt.savefig(f'PDF_{file_name}.png', dpi=300, bbox_inches='tight')



# def plot_pdf(data_dict, title, norm_dist=False, laplace_dist=False, 
#              log_Y=True, pdf_min=None, pdf_max=None, x_min=None, x_max=None, 
#              save=True, file_name=None, my_color=None, ticks_min=None, ticks_max=None):
#     """
#     Plot the Probability Density Function (PDF) of multiple datasets on top of each other.

#     Parameters:
#     - data_dict: dict, keys are dataset names and values are numpy arrays (matrices).
#     - title: str, title of the plot.
#     - norm_dist: bool, if True, overlays a normal distribution fit for each dataset.
#     - laplace_dist: bool, if True, overlays a Laplace distribution fit for each dataset.
#     - log_Y: bool, if True, sets the y-axis to logarithmic scale.
#     """
#     fontsize = 32
#     num_bins = 100
#     plt.figure(figsize=(10, 6))
#     chart_type = 'line'
#     i = 0
#     if my_color is None: color = ['black', 'blue']
#     else: color = my_color

#     for name, data in data_dict.items():
#         data_flat = data.flatten()
#         hist, bins = np.histogram(data_flat, bins=num_bins, density=True)
#         bin_centers = (bins[:-1] + bins[1:]) / 2

#         if chart_type == 'bar':
#             plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.5, label=name, edgecolor='black')
#         elif chart_type == 'line':
#             plt.plot(bin_centers, hist, label=name, color=color[i], linewidth=2)
#             i += 1

#         # Normal distribution fit
#         if norm_dist:
#             mu, sigma = np.mean(data_flat), np.std(data_flat)
#             x = np.linspace(min(data_flat), max(data_flat), num_bins)
#             pdf = norm.pdf(x, mu, sigma)
#             plt.plot(x, pdf, linestyle='dashed', color='red', label=f'{name} (Normal Fit)')
#             stats_text = f"{name} Normal Fit:   μ = {mu:.4f},   σ = {sigma:.4f}\n"
#             plt.figtext(0.5, -0.2, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
#             norm_dist = False

#         # Laplace distribution fit
#         elif laplace_dist:
#             loc, scale = laplace.fit(data_flat)
#             x = np.linspace(min(data_flat), max(data_flat), num_bins)
#             pdf_laplace = laplace.pdf(x, loc=loc, scale=scale)
#             plt.plot(x, pdf_laplace, linestyle='dashed', color='red', label=f'{name} (Laplace Fit)')
#             stats_text = f"{name} Laplace Fit:   μ = {loc:.4f},   b = {scale:.4f}\n"
#             plt.figtext(0.5, -0.2, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
#             laplace_dist = False

#     # Labels and formatting
#     plt.title(f'PDF of {title}', fontsize=fontsize, pad=70)
#     plt.xlabel(f'{title}', fontsize=fontsize, labelpad=20)
#     plt.ylabel('PDF', fontsize=fontsize, labelpad=40)
#     if log_Y: plt.yscale('log')
#     if pdf_min or pdf_max: 
#         plt.ylim(pdf_min, pdf_max)
#         if log_Y: plt.yticks(np.logspace(np.log10(pdf_min), np.log10(pdf_max), 3), fontsize=fontsize)
#         else: plt.yticks(np.linspace(pdf_min, pdf_max, 3), fontsize=fontsize)
#     if x_min or x_max:
#         if (ticks_min and ticks_max) is None:
#             ticks_min = x_min
#             ticks_max = x_max
#         plt.xlim(x_min, x_max)
#         plt.xticks(np.linspace(ticks_min, ticks_max, 3), fontsize=fontsize)

#     # plt.legend(loc='best')
#     # plt.grid(True)
#     if file_name is None: file_name = title
#     if save: plt.savefig(f'PDF_{file_name}_paper.pdf', dpi=300, bbox_inches='tight')


def plot_pdf(data_dict, title, norm_dist=False, laplace_dist=False, 
             log_Y=True, pdf_min=None, pdf_max=None, x_min=None, x_max=None, 
             save=True, file_name=None, my_color=None):
    """
    Plot the Probability Density Function (PDF) of multiple datasets on top of each other.
    """

    fontsize = 32
    num_bins = 100
    plt.figure(figsize=(10, 6))
    chart_type = 'line'
    i = 0
    if my_color is None: color = ['black', 'blue']
    else: color = my_color

    for name, data in data_dict.items():
        data_flat = data.flatten()
        hist, bins = np.histogram(data_flat, bins=num_bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if chart_type == 'bar':
            plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.5, label=name, edgecolor='black')
        elif chart_type == 'line':
            plt.plot(bin_centers, hist, label=name, color=color[i], linewidth=2)
            i += 1

        # Normal distribution fit
        if norm_dist:
            mu, sigma = np.mean(data_flat), np.std(data_flat)
            x = np.linspace(min(data_flat), max(data_flat), num_bins)
            pdf = norm.pdf(x, mu, sigma)
            # plt.plot(x, pdf, linestyle='dashed', color='red', label=rf'{name} (Normal Fit)')
            plt.plot(x, pdf, linestyle='-', color='red', label=rf'{name} (Normal Fit)')
            stats_text = rf'{name} Normal Fit: $\mu = {mu:.4f}$, $\sigma = {sigma:.4f}$'
            plt.figtext(0.5, -0.2, stats_text, ha='center', fontsize=fontsize-20,
                        bbox=dict(facecolor='white', alpha=0.5))
            norm_dist = False

        # Laplace distribution fit
        elif laplace_dist:
            loc, scale = laplace.fit(data_flat)
            x = np.linspace(min(data_flat), max(data_flat), num_bins)
            pdf_laplace = laplace.pdf(x, loc=loc, scale=scale)
            # plt.plot(x, pdf_laplace, linestyle='dashed', color='red', label=rf'{name} (Laplace Fit)')
            plt.plot(x, pdf_laplace, linestyle='-', color='red', label=rf'{name} (Laplace Fit)')
            stats_text = rf'{name} Laplace Fit: $\mu = {loc:.4f}$, $b = {scale:.4f}$'
            plt.figtext(0.5, -0.2, stats_text, ha='center', fontsize=fontsize-20,
                        bbox=dict(facecolor='white', alpha=0.5))
            laplace_dist = False
    
    # Labels and formatting
    plt.title(rf'PDF of {title}', fontsize=fontsize, pad=70)
    plt.xlabel(rf'{title}', fontsize=fontsize, labelpad=20)
    plt.ylabel(r'PDF', fontsize=fontsize, labelpad=40)
    
    ax = plt.gca()
    
    # Set axis scale before applying limits
    if log_Y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0, labelOnlyBase=True))
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))

    # Apply limits AFTER setting scale
    if pdf_min is not None and pdf_max is not None:
        ax.set_ylim(pdf_min, pdf_max)
    elif pdf_min is not None:
        ax.set_ylim(bottom=pdf_min)
    elif pdf_max is not None:
        ax.set_ylim(top=pdf_max)

    if x_min is not None or x_max is not None:
        if x_min is not None and x_max is not None:
            plt.xlim(x_min, x_max)
        elif x_min is not None:
            plt.xlim(left=x_min)
        elif x_max is not None:
            plt.xlim(right=x_max)

    # Tick label logic
    yticks = ax.get_yticks()
    # ax.set_yticks(yticks)
    if log_Y:
        yticklabels = [rf"$10^{{{int(np.log10(tick))}}}$" if i % 2 == 0 else "" for i, tick in enumerate(yticks)]
    else:
        yticklabels = [f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(yticks)]
    ax.set_yticklabels(yticklabels)

    # x-axis always plain
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='x')

    xticks = ax.get_xticks()
    # ax.set_xticks(xticks)
    xticklabels = [f"{tick:.0f}" if i % 2 == 0 else "" for i, tick in enumerate(xticks)]
    ax.set_xticklabels(xticklabels)

    # Tick and border styles
    ax.tick_params(width=2.5, length=6, direction='in')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    bold_font = FontProperties(weight='bold', size=fontsize-4)
    ax.set_xticklabels(xticklabels, fontproperties=bold_font)
    ax.set_yticklabels(yticklabels, fontproperties=bold_font)

    ax.legend(fontsize=fontsize-16, loc='best')
    if file_name is None: file_name = title
    if save:
        plt.savefig(f'PDF_{file_name}_paper.pdf', dpi=300, bbox_inches='tight')
        tikzplotlib.save(f'PDF_{file_name}.tex')

    plt.show()


def plot_contour(u, v_min=None, v_max=None, cutoff=None, c_map='coolwarm'):
    """
    Plots a 2D contour (imshow) of the middle z-slice of a 3D field `u`.

    Parameters:
    - u: 3D numpy array of shape (Nx, Ny, Nz)
    - cutoff: float or None, used in the title to distinguish DNS vs cutoff result
    - c_map: string, name of the colormap to use (default: 'coolwarm')
    """
    # Determine the z-slice index (middle)
    k = u.shape[2] // 2
    fontsize = 42
    # Set vmin and vmax for consistent color scaling
    if v_min is None:
        v_min = np.min(u[:, :, k])
        v_max = np.max(u[:, :, k])

    # Axis ticks
    tick_positions_x = [0, np.pi, 2*np.pi]
    tick_positions_y = [0, np.pi, 2*np.pi]
    tick_labels_x = ['0', r'$\pi$', r'$2\pi$']
    tick_labels_y = ['0', r'$\pi$', r'$2\pi$']

    # Plot
    plt.figure(figsize=(8, 6))
    img = plt.imshow(u[:, :, k],
                     origin='lower',
                     cmap=c_map,
                     vmin=v_min,
                     vmax=v_max,
                     extent=[0, 2*np.pi, 0, 2*np.pi])

    # Title
    title_text = (fr'u velocity - Cutoff {cutoff} ($z = \pi$)' if cutoff is not None
                  else r'u velocity - DNS ($z = \pi$)')
    plt.title(title_text, pad=30, fontsize=fontsize)

    # Axis labels
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)

    # Axis ticks
    plt.xticks(tick_positions_x, tick_labels_x, fontsize=fontsize)
    plt.yticks(tick_positions_y, tick_labels_y, fontsize=fontsize)

    # Colorbar
    cbar = plt.colorbar(img)
    cbar.set_label('u', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Save figure
    label = f'cutoff_{cutoff}' if cutoff is not None else 'DNS'
    plt.savefig(f'Middle_XY_Plane_u_{label}_{c_map}.pdf', dpi=300, bbox_inches='tight')
    plt.show()



def plot_scatter(u, u_comp, cutoff, title='Velocity u', label='u', file_name='Velocity_u'
                 , save=True, legend_flag=True, alpha=0.8):
    """
    Plots a scatter plot comparing true and compressed values.
    
    Parameters:
    - u: numpy array, ground truth velocity field
    - u_comp: numpy array, compressed or approximated velocity field
    - cutoff: float, cutoff used for compression (for labeling)
    - title: str, title of the plot
    - label: str, base label for axis and legend
    - file_name: str, filename for saving outputs (no extension)
    - save: bool, whether to save as .pdf and .tex
    """
    # Flatten the arrays for 1D scatter comparison
    u_flat = u.flatten()
    u_comp_flat = u_comp.flatten()

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u_comp_flat[0], u_flat[0], s=1, alpha=1, label=f'cutoff = {cutoff:.0e}', color='blue')
    ax.scatter(u_comp_flat[1:], u_flat[1:], s=1, alpha=alpha, color='blue')
    # Styling
    u_min_plt = (u_flat.min()<0)*1.05*u_flat.min() + (u_flat.min()>=0)*0.95*u_flat.min()
    u_max_plt = (u_flat.max()<0)*0.95*u_flat.max() + (u_flat.max()>=0)*1.05*u_flat.max()
    ax.plot([u_min_plt, u_max_plt], [u_min_plt, u_max_plt], 'black', linewidth=1, label='y = x')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f'{label} MPS', fontsize=18)
    ax.set_ylabel(f'{label} DNS', fontsize=18)
    if legend_flag: ax.legend(fontsize=14, loc='upper left')
    ax.set_aspect('equal', 'box')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.set_xlim(u_min_plt, u_max_plt)
    ax.set_ylim(u_min_plt, u_max_plt)
    # Save
    if save:
        plt.savefig(f'Scatter_{file_name}.png', dpi=300, bbox_inches='tight')
        # tikzplotlib.save(f'{file_name}.tex')
    plt.show()
