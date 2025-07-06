from typing import Optional

import numpy as np
import xarray as xr

from ..utils import CONFIGS


class PlotResponseBase:
    def __init__(
        self,
        model_info_steps: dict[str, xr.DataArray],
        resp_step: xr.Dataset,
        model_update: bool,
        nodal_resp_steps: Optional[xr.Dataset] = None,
    ):
        self.ModelInfoSteps = model_info_steps
        self.RespSteps = resp_step
        self.ModelUpdate = model_update
        self.nodal_resp_steps = nodal_resp_steps
        self.time = self.RespSteps.coords["time"].values
        self.num_steps = len(self.time)

        self.points_origin = self._get_node_da(0).to_numpy()
        self.bounds = self._get_node_da(0).attrs["bounds"]
        self.max_bound_size = self._get_node_da(0).attrs["maxBoundSize"]
        self.min_bound_size = self._get_node_da(0).attrs["minBoundSize"]
        model_dims = self._get_node_da(0).attrs["ndims"]
        # # show z-axis in 3d view
        self.show_zaxis = not np.max(model_dims) <= 2
        # ------------------------------------------------------------
        self.pargs = None
        self.resp_step = None  # response data
        self.resp_type = None  # response type
        self.component = None  # component to be visualized
        self.fiber_point = None  # fiber point for shell fiber response
        self.unit_symbol = ""  # unit symbol
        self.unit_factor = 1.0
        self.clim = (0, 1)  # color limits

        self.defo_scale_factor = None
        self.defo_coords = None  # deformed coordinates

        self.PKG_NAME = self.pkg_name = CONFIGS.get_pkg_name()

    def set_unit(self, symbol: Optional[str] = None, factor: Optional[float] = None):
        # unit
        if symbol is not None:
            self.unit_symbol = symbol
        if factor is not None:
            self.unit_factor = factor

    def _get_model_da(self, key, idx):
        dims = self.ModelInfoSteps[key].dims
        if self.ModelUpdate:
            da = self.ModelInfoSteps[key].isel(time=idx)
            da = da.dropna(dim=dims[1], how="any")
        else:
            da = self.ModelInfoSteps[key].isel(time=0)
        # tags = da.coords[dims[1]].values
        return da.copy()

    def _get_node_da(self, idx):
        nodal_data = self._get_model_da("NodalData", idx)
        unused_node_tags = nodal_data.attrs["unusedNodeTags"]
        if len(unused_node_tags) > 0:
            nodal_data = nodal_data.where(~nodal_data.coords["nodeTags"].isin(unused_node_tags), drop=True)
        return nodal_data

    def _get_line_da(self, idx):
        return self._get_model_da("AllLineElesData", idx)

    def _get_unstru_da(self, idx):
        return self._get_model_da("UnstructuralData", idx)

    def _get_bc_da(self, idx):
        return self._get_model_da("FixedNodalData", idx)

    def _get_mp_constraint_da(self, idx):
        return self._get_model_da("MPConstraintData", idx)

    def _get_resp_da(self, time_idx, resp_type, component=None):
        dims = self.RespSteps[resp_type].dims
        da = self.RespSteps[resp_type].isel(time=time_idx).copy()
        if self.ModelUpdate:
            da = da.dropna(dim=dims[1], how="all")
        if da.ndim == 1 or component is None:
            return da * self.unit_factor
        elif da.ndim == 2:
            return da.loc[:, component] * self.unit_factor
        elif da.ndim == 3:
            return da.loc[:, :, component] * self.unit_factor
        return None

    def _get_disp_da(self, idx):
        if self.nodal_resp_steps is None:
            data = self._get_resp_da(idx, "disp", ["UX", "UY", "UZ"])
        else:
            data = self.nodal_resp_steps["disp"].isel(time=idx).copy()
            if self.ModelUpdate:
                data = data.dropna(dim="nodeTags", how="all")
            data = data.sel(DOFs=["UX", "UY", "UZ"])
        return data / self.unit_factor  # come back to original unit

    def _set_defo_scale_factor(self, alpha=1.0):
        if self.defo_scale_factor is not None:
            return

        defos, poss = [], []
        for i in range(self.num_steps):
            defo = self._get_disp_da(i)
            pos = self._get_node_da(i)
            pos.coords["time"] = defo.coords["time"]
            defos.append(defo)
            poss.append(pos)

        if isinstance(alpha, str) or alpha is True:
            if self.ModelUpdate:
                scalars = [np.linalg.norm(resp, axis=-1) for resp in defos]
                scalars = [np.max(scalar) for scalar in scalars]
            else:
                scalars = np.linalg.norm(defos, axis=-1)
            maxv = np.max(scalars)
            alpha_ = 0.0 if maxv == 0 else self.max_bound_size * self.pargs.scale_factor / maxv
            alpha_ = alpha_
        elif alpha is False or alpha is None:
            alpha_ = 1.0
        elif isinstance(alpha, (int, float)):
            alpha_ = alpha
        self.defo_scale_factor = alpha_

        if self.ModelUpdate:
            defo_coords = [alpha_ * np.array(defo) + np.array(pos) for defo, pos in zip(defos, poss)]
            defo_coords = [
                xr.DataArray(coords, dims=pos.dims, coords=pos.coords) for coords, pos in zip(defo_coords, poss)
            ]
        else:
            poss_da = xr.concat(poss, dim="time")
            defo_coords = alpha_ * np.array(defos) + np.array(poss)
            defo_coords = xr.DataArray(defo_coords, dims=poss_da.dims, coords=poss_da.coords)
        self.defo_coords = defo_coords

    def _get_defo_coord_da(self, step, alpha):
        if not isinstance(alpha, bool) and alpha == 0.0:
            original_coords_da = self._get_node_da(step)
            return original_coords_da
        self._set_defo_scale_factor(alpha=alpha)
        node_deform_coords = self.defo_coords[step] if self.ModelUpdate else self.defo_coords.isel(time=step)
        return node_deform_coords

    @staticmethod
    def _get_line_cells(line_data):
        if len(line_data) > 0:
            line_cells = line_data.to_numpy().astype(int)
            line_tags = line_data.coords["eleTags"]
        else:
            line_cells, line_tags = [], []
        return line_cells, line_tags

    @staticmethod
    def _get_unstru_cells(unstru_data):
        if len(unstru_data) > 0:
            unstru_tags = unstru_data.coords["eleTags"]
            unstru_cell_types = np.array(unstru_data[:, -1], dtype=int)
            unstru_cells = unstru_data.to_numpy()
            if not np.any(np.isnan(unstru_cells)):
                unstru_cells_new = unstru_cells[:, :-1].astype(int)
            else:
                unstru_cells_new = []
                for cell in unstru_cells:
                    num = int(cell[0])
                    data = [num] + [int(data) for data in cell[1 : 1 + num]]
                    unstru_cells_new.extend(data)
        else:
            unstru_tags, unstru_cell_types, unstru_cells_new = [], [], []
        return unstru_tags, unstru_cell_types, unstru_cells_new

    def _dropnan_by_time(self, da):
        dims = da.dims
        time_dim = dims[0]
        cleaned_dataarrays = []
        for t in range(da.sizes[time_dim]):
            da_2d = da.isel({time_dim: t})
            if da_2d.size == 0 or any(dim == 0 for dim in da_2d.shape):
                cleaned_dataarrays.append([])
            else:
                dim2 = dims[1]
                da_2d_cleaned = da_2d.dropna(dim=dim2, how="any") if self.ModelUpdate else da_2d
                cleaned_dataarrays.append(da_2d_cleaned)
        return cleaned_dataarrays

    def _plot_outline(self, *args, **kwargs):
        pass

    def _plot_bc(self, *args, **kwargs):
        pass

    def _plot_bc_update(self, *args, **kwargs):
        pass

    def _plot_mp_constraint(self, *args, **kwargs):
        pass

    def _plot_mp_constraint_update(self, *args, **kwargs):
        pass

    def _plot_all_mesh(self, *args, **kwargs):
        pass

    def _update_plotter(self, *args, **kwargs):
        pass
