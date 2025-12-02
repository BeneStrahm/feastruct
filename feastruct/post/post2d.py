import numpy as np
import matplotlib.pyplot as plt
import feastruct.fea.bcs as bcs


class PostProcessor2D:
    """Class for post processing methods for 2D analyses.

    This class provides some post-processing methods for a particular 2D analysis that can be used
    to visualise tthe structural geometry and the finite element analysis results.

    :cvar analysis: Analysis object for post-processing
    :vartype analysis: :class:`~feastruct.fea.fea.fea`
    :cvar int n_subdiv: Number of subdivisions (total nodes) used to discretise frame elements in
        post-processing, such that higher order shape functions can be realised
    """

    def __init__(self, analysis, n_subdiv=11):
        """Inits the fea class.

        :param analysis: Analysis object for post-processing
        :type analysis: :class:`~feastruct.fea.fea.FiniteElementAnalysis`
        :param int n_subdiv: Number of subdivisions used to discretise frame elements in
            post-processing
        """

        self.analysis = analysis
        self.n_subdiv = n_subdiv

    def plot_geom(self, analysis_case, ax=None, fig=None, axis=[False, False], opt_results=None, supports=True, loads=True, undeformed=True, deformed=False, def_scale=1, dashed=False, showPlt=False, style_dict={}):
        """Method used to plot the structural mesh in the undeformed and/or deformed state. If no
        axes object is provided, a new axes object is created. N.B. this method is adapted from the
        MATLAB code by F.P. van der Meer: plotGeom.m.

        :param analysis_case: Analysis case
        :type analysis_case: :class:`~feastruct.fea.cases.AnalysisCase`
        :param ax: Axes object on which to plot
        :type ax: :class:`matplotlib.axes.Axes`
        :param fig: Figure object on which to plot
        :type fig: :class:`matplotlib
        :param [bool, bool] axis: list with bools whether or not the [x / y] axis is plotted
        :param opt_results: The results of the optimization. Defaults to None.
        :type opt_results: :class:`~designforreuse.opt.results.Results`
        :param bool supports: Whether or not the freedom case supports are rendered
        :param bool loads: Whether or not the load case loads are rendered
        :param bool undeformed: Whether or not the undeformed structure is plotted
        :param bool deformed: Whether or not the deformed structure is plotted
        :param float def_scale: Deformation scale used for plotting the deformed structure
        :param bool dashed: Whether or not to plot the structure with dashed lines if only the
            undeformed structure is to be plotted
        :param bool showPlt: Whether or not to show the plot
        """

        if ax is None:
            (fig, ax) = plt.subplots()

        # Load / apply mpl style
        from pyLEK.plotters.plotStyle import mplStyle

        # Find plot styles
        mplPath = mplStyle.findPlotStyle('feastruct')

        # Get the plot styles
        mplStyle.retrievePlotStyle(style_dict, mplPath)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(axis[0])
        ax.get_yaxis().set_visible(axis[1])

        # Add segment markers
        if opt_results is not None:
            phi = opt_results.phi[:, :, opt_results.k]
            for i, row in enumerate(phi):
                if np.any(row):
                    # Find the nodes at the start and end of the segment
                    i_start, i_end = np.nonzero(row)[0][[0, -1]]
                    start, end = self.analysis.elements[i_start].nodes[0], self.analysis.elements[i_end].nodes[1]

                    # Plot the segment marker
                    ax.plot([start.x, end.x], [start.y, end.y], linestyle='-', marker='|', markersize=5,
                            markeredgewidth=0.7, color='k', linewidth=0)

        for el in self.analysis.elements:
            if deformed:
                el.plot_deformed_element(
                    ax=ax, analysis_case=analysis_case, n_subdiv=self.n_subdiv, def_scale=def_scale)
                if undeformed:
                    el.plot_element(ax=ax, linestyle='--',
                                    linewidth=1, marker='')
            else:
                if dashed:
                    el.plot_element(ax=ax, linestyle='--',
                                    linewidth=1, marker='')
                else:
                    el.plot_element(ax=ax)

         # set initial plot limits
        if opt_results is not None:
            xmax = max(opt_results.L)
            (xmin, _, ymin, ymax, _, _) = self.analysis.get_node_lims()
        else:
            (xmin, xmax, ymin, ymax, _, _) = self.analysis.get_node_lims()

        ax.set_xlim(xmin-1e-12, xmax)
        ax.set_ylim(ymin-1e-10, ymax)

        if axis[0] == True:
            # Setup ticks to be at every 2nd element
            ax.set_xticks([xmin, xmax])

        # get 2% of the maxmimum dimension
        small = 0.025 * max(xmax-xmin, ymax-ymin)

        if supports:
            # generate list of supports and imposed displacements for unique nodes
            node_supports = []
            node_imposed_disps = []
            max_disp = 0

            # loop through supports
            for support in analysis_case.freedom_case.items:
                # if there is no imposed displacement, the support is fixed
                if support.val == 0:
                    # check to see if the node hasn't already been added
                    if support.node not in node_supports:
                        node_supports.append(support)

                # if there is an imposed displacement
                else:
                    node_imposed_disps.append(support)
                    if support.dof in [0, 1]:
                        max_disp = max(max_disp, abs(support.val))

        if supports:
            # plot supports
            for support in node_supports:
                support.plot_support(
                    ax=ax, small=small, get_support_angle=self.get_support_angle,
                    analysis_case=analysis_case, deformed=deformed, def_scale=def_scale)

            # plot imposed displacements
            for imposed_disp in node_imposed_disps:
                if imposed_disp.dof in [0, 1]:
                    imposed_disp.plot_imposed_disp(
                        ax=ax, max_disp=max_disp, small=small,
                        get_support_angle=self.get_support_angle, analysis_case=analysis_case,
                        deformed=deformed, def_scale=def_scale)
                elif imposed_disp.dof == 5:
                    imposed_disp.plot_imposed_rot(
                        ax=ax, small=small, get_support_angle=self.get_support_angle,
                        analysis_case=analysis_case, deformed=deformed, def_scale=def_scale)

        if loads:
            # find max force
            max_force = 0

            for load in analysis_case.load_case.items:
                if load.dof in [0, 1]:
                    max_force = max(max_force, abs(load.val))

            # plot loads
            for load in analysis_case.load_case.items:
                load.plot_load(
                    ax=ax, max_force=max_force, small=small,
                    get_support_angle=self.get_support_angle, analysis_case=analysis_case,
                    deformed=deformed, def_scale=def_scale)

        # plot layout
        # plt.axis('tight')
        ax.set_xlim(self.wide_lim((xmin, xmax)))
        if ymin != ymax:
            ax.set_ylim(self.wide_lim((ymin, ymax)))

        limratio = np.diff(ax.get_ylim())/np.diff(ax.get_xlim())

        if limratio < 0.5:
            ymid = np.mean(ax.get_ylim())
            ax.set_ylim(ymid + (ax.get_ylim() - ymid) * 0.5 / limratio)
        elif limratio > 1:
            xmid = np.mean(ax.get_xlim())
            ax.set_xlim(xmid + (ax.get_xlim() - xmid) * limratio)

        ax.set_aspect(1)
        plt.box(on=None)

        # # center the plot in the figure
        # if fig is not None:
        #     fig.canvas.draw()  # Ensure all artists are drawn first
        #     # Get the current axes position in figure coordinates
        #     ax_pos = ax.get_position()
        #     # Calculate centered position
        #     new_width = ax_pos.width
        #     new_left = (1 - new_width) / 2  # Center horizontally
        #     # Set new position (keeping vertical position unchanged)
        #     ax.set_position([new_left, ax_pos.y0, new_width, ax_pos.height])

        if showPlt == True:
            plt.show()

        return ax, fig

    def plot_reactions(self, analysis_case):
        """Method used to generate a plot of the reaction forces.

        :param analysis_case: Analysis case
        :type analysis_case: :class:`~feastruct.fea.cases.AnalysisCase`
        """

        (fig, ax) = plt.subplots()

        # get size of structure
        (xmin, xmax, ymin, ymax, _, _) = self.analysis.get_node_lims()

        # determine maximum reaction force
        max_reaction = 0

        for support in analysis_case.freedom_case.items:
            if support.dof in [0, 1]:
                reaction = support.get_reaction(analysis_case=analysis_case)
                max_reaction = max(max_reaction, abs(reaction))

        small = 0.025 * max(xmax-xmin, ymax-ymin)

        # plot reactions
        for support in analysis_case.freedom_case.items:
            support.plot_reaction(
                ax=ax, max_reaction=max_reaction, small=small,
                get_support_angle=self.get_support_angle, analysis_case=analysis_case)

        # plot the undeformed structure
        self.plot_geom(analysis_case=analysis_case, ax=ax, supports=False)

    def plot_decorator(func):
        def wrapper(self, analysis_case, ax=None, fig=None, axis=[False, False],  opt_results=None, max_forces=None, axial=False, shear=False, moment=False, bending_stiffness=False, text_values=True, scale=0.1, showPlt=False, pass_opt_results=False):
            if ax is None:
                (fig, ax) = plt.subplots()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.get_xaxis().set_visible(axis[0])
            ax.get_yaxis().set_visible(axis[1])

            # get size of structure
            if opt_results is not None:
                xmax = max(opt_results.L)
                (xmin, _, ymin, ymax, _, _) = self.analysis.get_node_lims()
            else:
                pass
                # (xmin, xmax, ymin, ymax, _, _) = self.analysis.get_node_lims()

            # determine maximum forces
            max_axial = 0
            max_shear = 0
            max_moment = 0
            max_bending_stiffness = 0

            # Use max_forces to determine the maximum forces of all systems, to have one scale for all
            if max_forces is not None:
                max_axial = max_forces['axial']
                max_shear = max_forces['shear']
                max_moment = max_forces['moment']
                max_bending_stiffness = max_forces['bending_stiffness']

            # loop throuh each element to get max forces
            # for el in self.analysis.elements:
            #     if axial:
            #         (_, afd) = el.get_afd(
            #             n=self.n_subdiv, analysis_case=analysis_case)
            #         max_axial = max(max_axial, max(
            #             abs(min(afd)), abs(max(afd))))
            #     if shear:
            #         (_, sfd) = el.get_sfd(
            #             n=self.n_subdiv, analysis_case=analysis_case)
            #         max_shear = max(max_shear, max(
            #             abs(min(sfd)), abs(max(sfd))))
            #     if moment:
            #         (_, bmd) = el.get_bmd(
            #             n=self.n_subdiv, analysis_case=analysis_case)
            #         max_moment = max(max_moment, max(
            #             abs(min(bmd)), abs(max(bmd))))
            #     if bending_stiffness:
            #         eid = el.get_ei()
            #         max_bending_stiffness = max(max_bending_stiffness,
            #                                     abs(eid),)

            scale_axial = scale * \
                max(xmax - xmin, ymax - ymin) / max(max_axial, 1e-8)
            scale_shear = scale * \
                max(xmax - xmin, ymax - ymin) / max(max_shear, 1e-8)
            scale_moment = scale * \
                max(xmax - xmin, ymax - ymin) / max(max_moment, 1e-8)
            scale_bending_stiffness = scale * \
                max(xmax - xmin, ymax - ymin) / \
                max(max_bending_stiffness, 1e-8)

            # loop throgh each element to plot the forces
            i = 0
            for n, el in enumerate(self.analysis.elements):

                # For section plot, only plot vertical lines at start and beginning
                # of each segment
                startSegment = midSegment = endSegment = False

                # Check if element is in segment i
                if opt_results is not None:
                    phi = opt_results.phi[:, :, opt_results.k]
                    if np.sum(phi[i, :]) > 0:
                        # find smallest index of non-zero value
                        i_start = np.min(np.nonzero(phi[i, :]))

                        # find max. index of non-zero value
                        i_end = np.max(np.nonzero(phi[i, :]))

                        # Mid point of the element
                        midSegmentEven = True
                        if (i_start + i_end) % 2 == 0:  # Check if the midpoint index is even
                            midSegmentEven = True
                        else:
                            midSegmentEven = False
                        i_mid = int((i_start + i_end) / 2)

                        # if element is at start / beginning of segment, plot
                        # vertical line
                        if n == i_start:
                            startSegment = True
                        if n == i_mid:
                            midSegment = True
                        if n == i_end:
                            i += 1
                            endSegment = True

                if not pass_opt_results:
                    opt_results = None

                if axial:
                    el.plot_axial_force(
                        ax=ax, fig=fig, analysis_case=analysis_case, opt_results=opt_results, scalef=scale_axial, idx=n, n_subdiv=self.n_subdiv, text_values=text_values,)
                if shear:
                    el.plot_shear_force(
                        ax=ax, fig=fig, analysis_case=analysis_case, opt_results=opt_results, scalef=scale_shear, idx=n, n_subdiv=self.n_subdiv, text_values=text_values,)
                if moment:
                    el.plot_bending_moment(
                        ax=ax, fig=fig, analysis_case=analysis_case, opt_results=opt_results, scalef=scale_moment, idx=n, n_subdiv=self.n_subdiv, text_values=text_values, startSegment=startSegment, endSegment=endSegment, midSegment=midSegment, midSegmentEven=midSegmentEven)
                if bending_stiffness:
                    el.plot_bending_stiffness(
                        ax=ax, fig=fig, analysis_case=analysis_case, opt_results=opt_results, scalef=scale_bending_stiffness, idx=n, n_subdiv=self.n_subdiv, text_values=text_values, startSegment=startSegment, endSegment=endSegment, midSegment=midSegment, midSegmentEven=midSegmentEven)

            # plot the undeformed structure
            ax, fig = self.plot_geom(analysis_case=analysis_case, opt_results=opt_results,
                                     ax=ax, fig=fig, axis=axis, loads=False, showPlt=showPlt)

            return ax, fig

        return wrapper

    @plot_decorator
    def plot_frame_forces(self, analysis_case, ax=None, fig=None, axis=[False, False], opt_results=None, max_forces=None, axial=False, shear=False, moment=False, text_values=True, scale=0.1, showPlt=False, pass_opt_results=False):
        """Method used to plot internal frame actions resulting from the analysis case.

        :param analysis_case: Analysis case
        :type analysis_case: :class:`~feastruct.fea.cases.AnalysisCase`
        :param ax: Axes object on which to plot
        :type ax: :class:`matplotlib.axes.Axes`
        :param fig: Figure object on which to plot
        :type fig: :class:`matplotlib
        :param [bool, bool] axis: list with bools whether or not the [x / y] axis is plotted
        :param opt_results: The results of the optimization. Defaults to None.
        :type opt_results: :class:`~designforreuse.opt.results.Results`
        :param bool axial: Whether or not the axial force diagram is displayed
        :param bool shear: Whether or not the shear force diagram is displayed
        :param bool moment: Whether or not the bending moment diagram is displayed

        :param bool text_values: Whether or not the values of the internal forces are displayed
        :param float scale: Scale used for plotting internal force diagrams. Corresponds to the
            fraction of the window that the largest action takes up
        :param bool showPlt: Whether or not to show the plot
        """
        pass

    @plot_decorator
    def plot_frame_sections(self, analysis_case, ax=None, fig=None, axis=[False, False], opt_results=None, max_forces=None, axial=False, shear=False, moment=False, text_values=True, scale=0.1, showPlt=False, pass_opt_results=True):
        """Method used to plot frame sections resulting the element to section mapping.

        :param analysis_case: Analysis case
        :type analysis_case: :class:`~feastruct.fea.cases.AnalysisCase`
        :param sections: Element section vector with cross section resistance (on of the following: 'Vr', 'Mr')
        :type sections: numpy.ndarray
        :param ax: Axes object on which to plot
        :type ax: :class:`matplotlib.axes.Axes`
        :param fig: Figure object on which to plot
        :type fig: :class:`matplotlib
        :param [bool, bool] axis: list with bools whether or not the [x / y] axis is plotted
        :param opt_results: The results of the optimization. Defaults to None.
        :type opt_results: :class:`~designforreuse.opt.results.Results`
        :param bool axial: Whether or not the axial force diagram is displayed
        :param bool shear: Whether or not the shear force diagram is displayed
        :param bool moment: Whether or not the bending moment diagram is displayed

        :param bool text_values: Whether or not the values of the internal forces are displayed
        :param float scale: Scale used for plotting internal force diagrams. Corresponds to the
            fraction of the window that the largest action takes up
        :param bool showPlt: Whether or not to show the plot
        """
        pass

    def plot_buckling_results(self, analysis_case, buckling_mode=0):
        """Method used to plot a buckling eigenvector. The undeformed structure is plotted with a
        dashed line.

        :param analysis_case: Analysis case
        :type analysis_case: :class:`~feastruct.fea.cases.AnalysisCase`
        :param int buckling_mode: Buckling mode to plot
        """

        (fig, ax) = plt.subplots()

        # set initial plot limits
        (xmin, xmax, ymin, ymax, _, _) = self.analysis.get_node_lims()

        # determine max eigenvector displacement value (ignore rotation)
        max_v = 0

        # loop through all the elements
        for el in self.analysis.elements:
            (w, v_el) = el.get_buckling_results(
                analysis_case=analysis_case, buckling_mode=buckling_mode)

            max_v = max(max_v, abs(v_el[0, 0]), abs(
                v_el[0, 1]), abs(v_el[1, 0]), abs(v_el[1, 1]))

        # determine plot scale
        scale = 0.1 * max(xmax - xmin, ymax - ymin) / max_v

        # plot eigenvectors
        for el in self.analysis.elements:
            (_, v_el) = el.get_buckling_results(
                analysis_case=analysis_case, buckling_mode=buckling_mode)

            el.plot_deformed_element(
                ax=ax, analysis_case=analysis_case, n_subdiv=self.n_subdiv, def_scale=scale, u_el=v_el)

        # plot the load factor (eigenvalue)
        ax.set_title("Load Factor for Mode {:d}: {:.4e}".format(
            buckling_mode, w), size=10)

        # plot the undeformed structure
        self.plot_geom(analysis_case=analysis_case, ax=ax, dashed=True)

    def plot_frequency_results(self, analysis_case, frequency_mode=0):
        """Method used to plot a frequency eigenvector. The undeformed structure is plotted with a
        dashed line.

        :param analysis_case: Analysis case
        :type analysis_case: :class:`~feastruct.fea.cases.AnalysisCase`
        :param int frequency_mode: Frequency mode to plot
        """

        (fig, ax) = plt.subplots()

        # set initial plot limits
        (xmin, xmax, ymin, ymax, _, _) = self.analysis.get_node_lims()

        # determine max eigenvector displacement value (ignore rotation)
        max_v = 0

        # loop through all the elements
        for el in self.analysis.elements:
            (w, v_el) = el.get_frequency_results(
                analysis_case=analysis_case, frequency_mode=frequency_mode)

            max_v = max(max_v, abs(v_el[0, 0]), abs(
                v_el[0, 1]), abs(v_el[1, 0]), abs(v_el[1, 1]))

        # determine plot scale
        scale = 0.1 * max(xmax - xmin, ymax - ymin) / max_v

        # plot eigenvectors
        for el in self.analysis.elements:
            (_, v_el) = el.get_frequency_results(
                analysis_case=analysis_case, frequency_mode=frequency_mode)

            el.plot_deformed_element(
                ax=ax, analysis_case=analysis_case, n_subdiv=self.n_subdiv, def_scale=scale, u_el=v_el)

        # plot the frequency (eigenvalue)
        ax.set_title("Frequency for Mode {:d}: {:.4e} Hz".format(
            frequency_mode, w), size=10)

        # plot the undeformed structure
        self.plot_geom(analysis_case=analysis_case, ax=ax, dashed=True)

    def get_support_angle(self, node, prefer_dir=None):
        """Given a node object, returns the optimal angle to plot a support. Essentially finds the
        average angle of the connected elements and considers a preferred plotting direction. N.B.
        this method is adapted from the MATLAB code by F.P. van der Meer: plotGeom.m.

        :param node: Node object
        :type node: :class:`~feastruct.fea.node.node`
        :param int prefer_dir: Preferred direction to plot the support, where 0 corresponds to the
            x-axis and 1 corresponds to the y-axis
        """

        # find angles to connected elements
        phi = []
        num_el = 0

        # loop through each element in the mesh
        for el in self.analysis.elements:
            # if the current element is connected to the node
            if node in el.nodes:
                num_el += 1
                # loop through all the nodes connected to the element
                for el_node in el.nodes:
                    # if the node is not the node in question
                    if el_node is not node:
                        dx = [el_node.x - node.x, el_node.y - node.y]
                        phi.append(np.arctan2(dx[1], dx[0]) / np.pi * 180)

        phi.sort()
        phi.append(phi[0] + 360)
        i0 = np.argmax(np.diff(phi))
        angle = (phi[i0] + phi[i0+1]) / 2 + 180

        if prefer_dir is not None:
            if prefer_dir == 1:
                if max(np.sin([phi[i0] * np.pi / 180, phi[i0+1] * np.pi / 180])) > -0.1:
                    angle = 90
            elif prefer_dir == 0:
                if max(np.cos([phi[i0] * np.pi / 180, phi[i0+1] * np.pi / 180])) > -0.1:
                    angle = 0

        return (angle, num_el)

    def wide_lim(self, x):
        """Returns a tuple corresponding to the axis limits (x) stretched by 2% on either side.

        :param x: List containing axis limits e.g. [xmin, xmax]
        :type x: list[float, float]
        :returns: Stretched axis limits (x1, x2)
        :rtype: tuple(float, float)
        """

        x2 = max(x)
        x1 = min(x)
        dx = x2-x1
        f = 0.02

        return (x1 - f * dx, x2 + f * dx)
