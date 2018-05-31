import sys
import numpy as np
from matplotlib.patches import Polygon
from fea.exceptions import FEAInputError


class BoundaryCondition:
    """asdlkjaskld
    """

    def __init__(self, analysis, node_id, val, dir):
        # TODO: check value types e.g. node_id and dir are ints, check dir is
        # within dofs limits

        # find the node object corresponding to node_id
        try:
            node = analysis.find_node(node_id)
        except FEAInputError as error:
            print(error)
            sys.exit(1)

        self.node = node
        self.val = val
        self.dir = dir


class NodalSupport(BoundaryCondition):
    """asldkjasdl
    """

    def __init__(self, analysis, node_id, val, dir):
        super().__init__(analysis, node_id, val, dir)

        self.reaction = []

    def apply_support(self, K, f_ext):
        """sadsad
        """

        # modify stiffness matrix and f_ext
        K[self.node.dofs[self.dir-1], :] = 0
        K[self.node.dofs[self.dir-1], self.node.dofs[self.dir-1]] = 1
        f_ext[self.node.dofs[self.dir-1]] = self.val

    def get_reaction(self, case_id):
        """
        """

        # get dictionary reaction entry for given case_id
        reaction = next(d for d in self.reaction if d["case_id"] == case_id)

        if reaction is not None:
            return reaction["f_ext"]
        else:
            raise FEAInputError("""Cannot find an analysis result for
            case_id: {} at node_id: {}""".format(case_id, self.node_id))

    def plot_support(self, ax, max_disp, small, get_support_angle, case_id,
                     deformed, def_scale):
        """asdasdas
        """

        if self.node.fixity is not [1, 1, 0]:
            (angle, num_el) = get_support_angle(self.node)

        if self.node.fixity == [1, 0, 0]:
            # ploy a y-roller
            angle = round(angle / 180) * 180
            self.plot_xysupport(ax, angle, True, num_el == 1, small, case_id,
                                deformed, def_scale)

        elif self.node.fixity == [0, 1, 0]:
            # plot an x-roller
            if np.mod(angle + 1, 180) < 2:  # prefer support below
                angle = 90
            else:
                angle = round((angle + 90) / 180) * 180 - 90

            self.plot_xysupport(ax, angle, True, num_el == 1, small, case_id,
                                deformed, def_scale)

        elif self.node.fixity == [1, 1, 0]:
            # plot a hinge
            (angle, num_el) = get_support_angle(self.node, 2)
            self.plot_xysupport(ax, angle, False, num_el == 1, small, case_id,
                                deformed, def_scale)

        elif self.node.fixity == [0, 0, 1]:
            ax.plot(self.node.x, self.node.y, 'kx', markersize=8)

        else:
            # plot a support with moment fixity
            if self.node.fixity == [1, 1, 1]:
                # plot a fixed support
                s = np.sin(angle * np.pi / 180)
                c = np.cos(angle * np.pi / 180)
                rot_mat = np.array([[c, -s], [s, c]])
                line = np.array([[0, 0], [-1, 1]]) * small
                rect = np.array([[-0.6, -0.6, 0, 0], [-1, 1, 1, -1]]) * small
                ec = 'none'

            elif self.node.fixity == [1, 0, 1]:
                # plot y-roller block
                angle = round(angle / 180) * 180
                s = np.sin(angle * np.pi / 180)
                c = np.cos(angle * np.pi / 180)
                rot_mat = np.array([[c, -s], [s, c]])
                line = np.array([[-0.85, -0.85], [-1, 1]]) * small
                rect = np.array([[-0.6, -0.6, 0, 0], [-1, 1, 1, -1]]) * small
                ec = 'k'

            elif self.node.fixity == [0, 1, 1]:
                # plot x-roller block
                angle = round((angle + 90) / 180) * 180 - 90
                s = np.sin(angle * np.pi / 180)
                c = np.cos(angle * np.pi / 180)
                rot_mat = np.array([[c, -s], [s, c]])
                line = np.array([[-0.85, -0.85], [-1, 1]]) * small
                rect = np.array([[-0.6, -0.6, 0, 0], [-1, 1, 1, -1]]) * small
                ec = 'k'

            rot_line = np.matmul(rot_mat, line)
            rot_rect = np.matmul(rot_mat, rect)

            # add coordinates of node
            if deformed:
                # get displacement of node for current analysis case
                u = self.node.get_displacement(case_id)

                rot_line[0, :] += self.node.x + u[0] * def_scale
                rot_line[1, :] += self.node.y + u[1] * def_scale
                rot_rect[0, :] += self.node.x + u[0] * def_scale
                rot_rect[1, :] += self.node.y + u[1] * def_scale
            else:
                rot_line[0, :] += self.node.x
                rot_line[1, :] += self.node.y
                rot_rect[0, :] += self.node.x
                rot_rect[1, :] += self.node.y

            ax.plot(rot_line[0, :], rot_line[1, :], 'k-', linewidth=1)
            ax.add_patch(Polygon(np.transpose(rot_rect),
                                 facecolor=(0.7, 0.7, 0.7), edgecolor=ec))

    def plot_imposed_disp(self, ax, max_disp, small, get_support_angle,
                          case_id, deformed, def_scale):
        """aslkdjsak

        N.B. this method is adopted from the MATLAB code by F.P. van der Meer:
        plotGeom.m.
        """

        val = self.val / max_disp
        offset = 0.5 * small

        lf = abs(val) * 1.5 * small  # arrow length
        lh = 0.6 * small  # arrow head length
        wh = 0.6 * small  # arrow head width
        sp = 0.15 * small  # half spacing between double line
        lf = max(lf, lh * 1.5)

        (angle, num_el) = get_support_angle(self.node)
        s = np.sin(angle * np.pi / 180)
        c = np.cos(angle * np.pi / 180)
        n = np.array([c, s])
        inward = (n[self.dir-1] == 0 or np.sign(n[self.dir-1]) == np.sign(val))

        to_rotate = (self.dir - 1) * 90 + (n[self.dir-1] >= 0) * 180
        sr = np.sin(to_rotate * np.pi / 180)
        cr = np.cos(to_rotate * np.pi / 180)
        rot_mat = np.array([[cr, -sr], [sr, cr]])

        x0 = offset + inward * lf
        x2 = offset + (not inward) * lf
        x1 = x2 + (inward) * lh - (not inward) * lh
        pp = np.array([[x1, x1, x2], [-wh / 2, wh / 2, 0]])
        ll = np.array([[x1, x0, x0, x1], [sp, sp, -sp, -sp]])

        rl = np.matmul(rot_mat, ll)
        rp = np.matmul(rot_mat, pp)

        # add coordinates of node
        if deformed:
            # get displacement of node for current analysis case
            u = self.node.get_displacement(case_id)

            rp[0, :] += self.node.x + u[0] * def_scale
            rp[1, :] += self.node.y + u[1] * def_scale
            rl[0, :] += self.node.x + u[0] * def_scale
            rl[1, :] += self.node.y + u[1] * def_scale
        else:
            rp[0, :] += self.node.x
            rp[1, :] += self.node.y
            rl[0, :] += self.node.x
            rl[1, :] += self.node.y

        ax.plot(rl[0, :], rl[1, :], 'k-')
        ax.add_patch(Polygon(np.transpose(rp),
                             facecolor='none', linewidth=1, edgecolor='k'))

    def plot_imposed_rot(self, ax, small, get_support_angle, case_id, deformed,
                         def_scale):
        """aslkdjsak

        N.B. this method is adopted from the MATLAB code by F.P. van der Meer:
        plotGeom.m.
        """

        lh = 0.4 * small  # arrow head length
        wh = 0.4 * small  # arrow head width
        r1 = 1.0 * small
        r2 = 1.2 * small
        (angle, num_el) = get_support_angle(self.node)
        ths = np.arange(100, 261)

        s = np.sin(angle * np.pi / 180)
        c = np.cos(angle * np.pi / 180)
        rot_mat = np.array([[c, -s], [s, c]])

        # make arrow tail around (0,0)
        rr = (r1 + r2) / 2
        ll = np.array([rr * np.cos(ths * np.pi / 180),
                       rr * np.sin(ths * np.pi / 180)])
        l1 = np.array([r1 * np.cos(ths * np.pi / 180),
                       r1 * np.sin(ths * np.pi / 180)])
        l2 = np.array([r2 * np.cos(ths * np.pi / 180),
                       r2 * np.sin(ths * np.pi / 180)])

        # make arrow head at (0,0)
        pp = np.array([[-lh, -lh, 0], [-wh / 2, wh / 2, 0]])

        # rotate arrow head around (0,0)
        if self.val > 0:
            thTip = 90 - ths[11]
            xTip = ll[:, -1]
            l1 = l1[:, 1:-21]
            l2 = l2[:, 1:-21]
            ibase = 0
        else:
            thTip = ths[11] - 90
            xTip = ll[:, 0]
            l1 = l1[:, 21:]
            l2 = l2[:, 21:]
            ibase = np.shape(l1)[1] - 1

        cTip = np.cos(thTip * np.pi / 180)
        sTip = np.sin(thTip * np.pi / 180)
        rTip = np.array([[cTip, -sTip], [sTip, cTip]])
        pp = np.matmul(rTip, pp)

        # shift arrow head to tip
        pp[0, :] += xTip[0]
        pp[1, :] += xTip[1]

        # rotate arrow to align it with the node
        rl1 = np.matmul(rot_mat, l1)
        rl2 = np.matmul(rot_mat, l2)
        rp = np.matmul(rot_mat, pp)

        # add coordinates of node
        if deformed:
            # get displacement of node for current analysis case
            u = self.node.get_displacement(case_id)

            rp[0, :] += self.node.x + u[0] * def_scale
            rp[1, :] += self.node.y + u[1] * def_scale
            rl1[0, :] += self.node.x + u[0] * def_scale
            rl1[1, :] += self.node.y + u[1] * def_scale
            rl2[0, :] += self.node.x + u[0] * def_scale
            rl2[1, :] += self.node.y + u[1] * def_scale
        else:
            rp[0, :] += self.node.x
            rp[1, :] += self.node.y
            rl1[0, :] += self.node.x
            rl1[1, :] += self.node.y
            rl2[0, :] += self.node.x
            rl2[1, :] += self.node.y

        # shift arrow to node and plot
        ax.plot(rl1[0, :], rl1[1, :], 'k-')
        ax.plot(rl2[0, :], rl2[1, :], 'k-')
        ax.plot(np.append(rl1[0, ibase], rl2[0, ibase]),
                np.append(rl1[1, ibase], rl2[1, ibase]), 'k-')
        ax.add_patch(Polygon(np.transpose(rp),
                             facecolor='none', linewidth=1, edgecolor='k'))

    def plot_reaction(self, ax, case_id, small, max_reaction,
                      get_support_angle):
        """
        """

        # get reaction force
        try:
            reaction = self.get_reaction(case_id)
        except FEAInputError as error:
            print(error)
            sys.exit(1)

        # dont plot small reaction
        if abs(reaction) < 1e-6:
            return

        if self.dir in (1, 2):
            val = reaction / max_reaction

            lf = abs(val) * 1.5 * small  # arrow length
            lh = 0.4 * small  # arrow head length
            wh = 0.4 * small  # arrow head width
            lf = max(lf, lh * 1.5)
            offset = 0.5 * small
            xoff = 0
            yoff = 0

            if self.dir == 1:
                rot_mat = np.array([[-1, 0], [0, -1]]) * np.sign(val)
                va = 'center'
                if val > 0:
                    ha = 'right'
                    xoff = -offset / 2
                else:
                    ha = 'left'
                    xoff = offset / 2
            elif self.dir == 2:
                rot_mat = np.array([[0, 1], [-1, 0]]) * np.sign(val)
                ha = 'center'
                if val > 0:
                    va = 'top'
                    yoff = -offset / 2
                else:
                    va = 'bottom'
                    yoff = offset / 2

            inward = True

            ll = np.array([[offset, offset + lf], [0, 0]])
            p0 = offset
            p1 = p0 + lh
            pp = np.array([[p1, p1, p0], [-wh / 2, wh / 2, 0]])

            # correct end of arrow line
            if inward:
                ll[0, 0] += lh
            else:
                ll[0, 1] -= lh

            rl = np.matmul(rot_mat, ll)
            rp = np.matmul(rot_mat, pp)
            rl[0, :] += self.node.x
            rl[1, :] += self.node.y
            rp[0, :] += self.node.x
            rp[1, :] += self.node.y
            s = 0
            e = None

            tl = np.array([rl[0, 1] + xoff, rl[1, 1] + yoff])
        else:
            (angle, num_el) = get_support_angle(self.node)
            s = np.sin(angle * np.pi / 180)
            c = np.cos(angle * np.pi / 180)

            lh = 0.3 * small  # arrow head length
            wh = 0.3 * small  # arrow head width
            rr = 1.5 * small
            ths = np.arange(100, 261)
            rot_mat = np.array([[c, -s], [s, c]])

            # make arrow tail around (0,0)
            ll = np.array([rr * np.cos(ths * np.pi / 180),
                           rr * np.sin(ths * np.pi / 180)])

            # make arrow head at (0,0)
            pp = np.array([[-lh, -lh, 0], [-wh / 2, wh / 2, 0]])

            # rotate arrow head around (0,0)
            if reaction > 0:
                thTip = 90 - ths[11]
                xTip = ll[:, -1]
                s = 0
                e = -1
            else:
                thTip = ths[11] - 90
                xTip = ll[:, 0]
                s = 1
                e = None

            cTip = np.cos(thTip * np.pi / 180)
            sTip = np.sin(thTip * np.pi / 180)
            rTip = np.array([[cTip, -sTip], [sTip, cTip]])
            pp = np.matmul(rTip, pp)

            # shift arrow head to tip
            pp[0, :] += xTip[0]
            pp[1, :] += xTip[1]

            # rotate arrow to align it with the node
            rl = np.matmul(rot_mat, ll)
            rp = np.matmul(rot_mat, pp)
            rl[0, :] += self.node.x
            rl[1, :] += self.node.y
            rp[0, :] += self.node.x
            rp[1, :] += self.node.y

            ha = 'center'
            va = 'center'
            tl = np.array([rl[0, 0], rl[1, 0]])

        ax.plot(rl[0, s:e], rl[1, s:e], linewidth=1.5, color='r')
        ax.add_patch(Polygon(np.transpose(rp), facecolor='r'))
        ax.text(tl[0], tl[1], "{:5.3g}".format(reaction), size=8,
                horizontalalignment=ha, verticalalignment=va)

    def plot_xysupport(self, ax, angle, roller, hinge, small, case_id,
                       deformed, def_scale):
        """aslkdjsak

        N.B. this method is adopted from the MATLAB code by F.P. van der Meer:
        plotGeom.m.
        """

        # determine coordinates of node
        if deformed:
            # get displacement of node for current analysis case
            u = self.node.get_displacement(case_id)

            x = self.node.x + u[0] * def_scale
            y = self.node.y + u[1] * def_scale
        else:
            x = self.node.x
            y = self.node.y

        # determine coordinates of triangle
        dx = small
        h = np.sqrt(3) / 2
        triangle = np.array([[-h, -h, -h, 0, -h], [-1, 1, 0.5, 0, -0.5]]) * dx
        s = np.sin(angle * np.pi / 180)
        c = np.cos(angle * np.pi / 180)
        rot_mat = np.array([[c, -s], [s, c]])
        rot_triangle = np.matmul(rot_mat, triangle)

        if roller:
            line = np.array([[-1.1, -1.1], [-1, 1]]) * dx
            rot_line = np.matmul(rot_mat, line)
            ax.plot(rot_line[0, :] + x, rot_line[1, :] + y, 'k-', linewidth=1)
        else:
            rect = np.array([[-1.4, -1.4, -h, -h], [-1, 1, 1, -1]]) * dx
            rot_rect = np.matmul(rot_mat, rect)
            rot_rect[0, :] += x
            rot_rect[1, :] += y
            ax.add_patch(Polygon(np.transpose(rot_rect),
                                 facecolor=(0.7, 0.7, 0.7)))

        ax.plot(rot_triangle[0, :] + x, rot_triangle[1, :] + y, 'k-',
                linewidth=1)

        if hinge:
            ax.plot(x, y, 'ko', markerfacecolor='w', linewidth=1, markersize=4)


class NodalLoad(BoundaryCondition):
    """asldkjasdl
    """

    def __init__(self, analysis, node_id, val, dir):
        super().__init__(analysis, node_id, val, dir)

    def apply_load(self, f_ext):
        """alskdjaskld
        """

        # add load to f_ext, selecting the correct dof from dofs
        f_ext[self.node.dofs[self.dir-1]] = self.val

    def plot_load(self, ax, max_force, small, get_support_angle, case_id,
                  deformed, def_scale):
        """aslkdjsak

        N.B. this method is adopted from the MATLAB code by F.P. van der Meer:
        plotGeom.m.
        """

        # determine coordinates of node
        if deformed:
            # get displacement of node for current analysis case
            u = self.node.get_displacement(case_id)

            x = self.node.x + u[0] * def_scale
            y = self.node.y + u[1] * def_scale
        else:
            x = self.node.x
            y = self.node.y

        val = self.val / max_force

        offset = 0.5 * small
        (angle, num_el) = get_support_angle(self.node)
        s = np.sin(angle * np.pi / 180)
        c = np.cos(angle * np.pi / 180)

        # plot nodal force
        if self.dir in (1, 2):
            lf = abs(val) * 1.5 * small  # arrow length
            lh = 0.6 * small  # arrow head length
            wh = 0.6 * small  # arrow head width
            lf = max(lf, lh * 1.5)

            n = np.array([c, s])
            inward = (n[self.dir-1] == 0 or
                      np.sign(n[self.dir-1]) == np.sign(val))

            to_rotate = (self.dir - 1) * 90 + (n[self.dir-1] > 0) * 180
            sr = np.sin(to_rotate * np.pi / 180)
            cr = np.cos(to_rotate * np.pi / 180)
            rot_mat = np.array([[cr, -sr], [sr, cr]])

            ll = np.array([[offset, offset + lf], [0, 0]])
            p0 = offset + (not inward) * lf
            p1 = p0 + (inward) * lh - (not inward) * lh
            pp = np.array([[p1, p1, p0], [-wh / 2, wh / 2, 0]])

            # correct end of arrow line
            if inward:
                ll[0, 0] += lh
            else:
                ll[0, 1] -= lh

            rl = np.matmul(rot_mat, ll)
            rp = np.matmul(rot_mat, pp)
            rp[0, :] += x
            rp[1, :] += y
            s = 0
            e = None

        # plot nodal moment
        else:
            lh = 0.4 * small  # arrow head length
            wh = 0.4 * small  # arrow head width
            rr = 1.5 * small
            ths = np.arange(100, 261)
            rot_mat = np.array([[c, -s], [s, c]])

            # make arrow tail around (0,0)
            ll = np.array([rr * np.cos(ths * np.pi / 180),
                           rr * np.sin(ths * np.pi / 180)])

            # make arrow head at (0,0)
            pp = np.array([[-lh, -lh, 0], [-wh / 2, wh / 2, 0]])

            # rotate arrow head around (0,0)
            if val > 0:
                thTip = 90 - ths[11]
                xTip = ll[:, -1]
                s = 0
                e = -1
            else:
                thTip = ths[11] - 90
                xTip = ll[:, 0]
                s = 1
                e = None

            cTip = np.cos(thTip * np.pi / 180)
            sTip = np.sin(thTip * np.pi / 180)
            rTip = np.array([[cTip, -sTip], [sTip, cTip]])
            pp = np.matmul(rTip, pp)

            # shift arrow head to tip
            pp[0, :] += xTip[0]
            pp[1, :] += xTip[1]

            # rotate arrow to align it with the node
            rl = np.matmul(rot_mat, ll)
            rp = np.matmul(rot_mat, pp)
            rp[0, :] += x
            rp[1, :] += y

        ax.plot(rl[0, s:e] + x, rl[1, s:e] + y, 'k-', linewidth=2)
        ax.add_patch(Polygon(np.transpose(rp), facecolor='k'))