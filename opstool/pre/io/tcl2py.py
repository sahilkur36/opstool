import tkinter

from rich import print as rprint


def tcl2py(
    input_file: str,
    output_file: str,
    prefix: str = "ops",
    encoding: str = "utf-8",
    keep_comments: bool = False,
):
    """Convert OpenSees ``Tcl`` code to OpenSeesPy ``Python`` format.

    .. Note::
        * This function supports ``Tcl`` syntax and will flatten your ``Tcl`` code, including ``loops``,
          ``judgments``, ``assignments``, ``proc``, etc.,

        * Do not use assignment statements for OpenSees commands, such as
          ``set ok [analyze 1]``, ``set lambdaN [eigen 10]``, it will trigger
          an error! This is because **this function does not run the OpenSees command at all**.

        * If an encoding error is reported, please use software such as vscode to re-save the file encoding as ``UTF-8`` in advance.

    Parameters
    ----------
    input_file : str
        The name of input ``.tcl`` file.
    output_file : str
        The name of output ``.py`` file.
    prefix : str, optional
        prefix name of openseespy, by default ``ops``.
        I.e., ``import openseespy.opensees as ops``.
        If None or void str '', the prefix is not used.
        I.e., ``from openseespy.opensees import *``.
    encoding: str, optional
        file encoding format, by default "utf-8".
    keep_comments: bool, optional
        Comments are preserved, by default False.
        Note that this parameter will replace all opensees commands in the comment line, if any.
    """
    if not input_file.endswith(".tcl"):
        input_file += ".tcl"
    if not output_file.endswith(".py"):
        output_file += ".py"
    if prefix:
        import_txt = f"import openseespy.opensees as {prefix}\n\n"
        prefix += "."
    else:
        import_txt = "from openseespy.opensees import *\n\n"
        prefix = ""
    if keep_comments:
        with open(input_file, encoding=encoding) as f:
            tcl_list = f.readlines()
        for i, src in enumerate(tcl_list):
            if src[0] == "#":
                tcl_list[i] = (
                    src.replace("###", "comments___ ")
                    .replace("##", "comments___ ")
                    .replace("#", "comments___ ")
                    .replace("$", "variable___ ")
                )
        tcl_src = "".join(tcl_list)
    else:
        with open(input_file, encoding=encoding) as f:
            tcl_src = f.read()
    tcl_src = tcl_src.replace("{", " { ")
    tcl_src = tcl_src.replace("}", " } ")
    tcl_src = tcl_src.replace("#", "# ")

    ops_interp = __OPSInterp(prefix, keep_comments=keep_comments)

    try:
        ops_interp.eval(tcl_src)
    finally:
        with open(output_file, mode="w", encoding=encoding) as fw:
            fw.write("# This file is created by opstool.tcl2py(), author:: Yexiang Yan\n\n")
            fw.write(import_txt)
            for line in ops_interp.get_opspy_cmds():
                fw.write(line + "\n")
    rprint(
        f"[bold #34bf49]OpenSeesPy[/bold #34bf49] file "
        f"[bold #d20962]{output_file}[/bold #d20962] has been created successfully!"
    )


def _type_convert(a):
    if isinstance(a, str):
        try:
            a = int(a)
        except ValueError:
            try:
                a = float(a)
            except ValueError:
                a = str(a)
    return a


def _remove_commit(args, obj="#"):
    if obj in args:
        idx = args.index(obj)
        args = args[:idx]
    return args


def _process_args(args, keep_comments=False):
    comments = ""
    if "#" in args:
        idx = args.index("#")
        if keep_comments:
            comments = " ".join(args[idx:])
            comments = "  " + comments
        args = args[:idx]
    args = tuple([_type_convert(i) for i in args])
    return args, comments


def _get_cmds(args, cmd, prefix, keep_comments=False):
    args, comments = _process_args(args, keep_comments=keep_comments)
    if len(args) == 1:
        if isinstance(args[0], str):
            return f"{prefix}{cmd}('{args[0]}'){comments}"
        else:
            return f"{prefix}{cmd}({args[0]}){comments}"
    else:
        return f"{prefix}{cmd}{args}{comments}"


class __OPSInterp:
    def __init__(self, prefix, keep_comments) -> None:
        self.prefix = prefix
        self.keep_comments = keep_comments
        self.interp = tkinter.Tcl()
        self.contents = []

    def _comments(self, *args):
        args = [src.replace("comments___", "#") for src in args]
        args = [src.replace("variable___", "$") for src in args]
        if args:
            args = " ".join(args).replace("# ", "#").replace("$ ", "$")
            self.contents.append(f"# {args}")
        else:
            self.contents.append("#")

    def _puts(self, *args):
        if len(args) == 1:
            self.contents.append(f"print('{args[0]}')")
        else:
            self.contents.append(f"print{args}")

    def _wipe(self, *args):
        src = _get_cmds(args, "wipe", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _model(self, *args):
        src = _get_cmds(args, "model", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _node(self, *args):
        src = _get_cmds(args, "node", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _fix(self, *args):
        src = _get_cmds(args, "fix", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _fixX(self, *args):
        src = _get_cmds(args, "fixX", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _fixY(self, *args):
        src = _get_cmds(args, "fixY", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _fixZ(self, *args):
        src = _get_cmds(args, "fixZ", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _equalDOF(self, *args):
        src = _get_cmds(args, "equalDOF", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _equalDOF_Mixed(self, *args):
        src = _get_cmds(args, "equalDOF_Mixed", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _rigidDiaphragm(self, *args):
        src = _get_cmds(args, "rigidDiaphragm", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _rigidLink(self, *args):
        src = _get_cmds(args, "rigidLink", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _uniaxialMaterial(self, *args):
        src = _get_cmds(args, "uniaxialMaterial", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nDMaterial(self, *args):
        src = _get_cmds(args, "nDMaterial", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _beamIntegration(self, *args):
        src = _get_cmds(args, "beamIntegration", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _section(self, *args):
        args, comments = _process_args(args, keep_comments=self.keep_comments)
        if args[0] in (
            "Fiber",
            "fiberSec",
            "FiberWarping",
            "FiberAsym",
            "FiberThermal",
            "NDFiber",
            "NDFiberWarping",
        ):
            if args[0] not in ["NDFiber", "NDFiberWarping"] and ("-GJ" not in args or "-torsion" not in args):
                rprint(
                    "[bold #d20962]Warning[/bold #d20962]: "
                    "-GJ or -torsion not used for fiber section, GJ=100000000 is assumed!"
                )
                new_args = (args[0], args[1], "-GJ", 1.0e8)
            else:
                new_args = args[:4]
            self.contents.append(f"{self.prefix}section{new_args}{comments}")
            txt = args[-1]
            txt.replace("\\n", "")
            self.interp.eval(txt)
        else:
            self.contents.append(f"{self.prefix}section{args}{comments}")

    def _fiber(self, *args):
        src = _get_cmds(args, "fiber", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _patch(self, *args):
        src = _get_cmds(args, "patch", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _layer(self, *args):
        src = _get_cmds(args, "layer", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _element(self, *args):
        args, comments = _process_args(args, keep_comments=self.keep_comments)
        if args[0] not in [
            "nonlinearBeamColumn",
            "forceBeamColumn",
            "dispBeamColumn",
            "forceBeamColumnCBDI",
            "forceBeamColumnCSBDI",
            "forceBeamColumnWarping",
            "forceBeamColumnThermal",
            "elasticForceBeamColumnWarping",
            "dispBeamColumnNL",
            "dispBeamColumnThermal",
            "nonlinearBeamColumn",
            "dispBeamColumnWithSensitivity",
        ]:
            self.contents.append(f"{self.prefix}element{args}{comments}")
        else:
            eleTag = args[1]
            secTag = args[5]
            if isinstance(secTag, int):
                Np = args[4]
                transfTag = args[6]
                if args[0] == "dispBeamColumn":
                    self.contents.append(f"{self.prefix}beamIntegration('Legendre', {eleTag}, {secTag}, {Np})")
                else:
                    self.contents.append(f"{self.prefix}beamIntegration('Lobatto', {eleTag}, {secTag}, {Np})")
                idx = 7
            elif secTag == "-sections":  # Handle variable section tags
                Np = args[4]
                sectags = args[6 : 6 + Np]
                transfTag = args[6 + Np]
                idx = 6 + Np + 1
                if args[0] == "dispBeamColumn":
                    self.contents.append(f"{self.prefix}beamIntegration('Legendre', {eleTag}, {Np}, *{sectags})")
                else:
                    self.contents.append(f"{self.prefix}beamIntegration('Lobatto', {eleTag}, {Np}, *{sectags})")
            else:
                transfTag = args[4]
                interp_paras = []
                idx = 6
                for i, arg in enumerate(args[6:]):
                    if not isinstance(arg, str):
                        interp_paras.append(arg)
                    else:
                        idx += i
                        break
                self.contents.append(f"{self.prefix}beamIntegration('{args[5]}', {eleTag}, *{interp_paras})")
            # write the element command
            if args[0] == "nonlinearBeamColumn":
                args[0] = "forceBeamColumn"
            if "-mass" not in args and "-iter" not in args and "-cMass" not in args:
                self.contents.append(
                    f"{self.prefix}element('{args[0]}', {eleTag}, {args[2]}, {args[3]}, {transfTag}, {eleTag}){comments}"
                )
            else:
                self.contents.append(
                    f"{self.prefix}element('{args[0]}', {eleTag}, {args[2]}, "
                    f"{args[3]}, {transfTag}, {eleTag}, *{args[idx:]}){comments}"
                )

    def _timeSeries(self, *args):
        args, comments = _process_args(args, keep_comments=self.keep_comments)
        if args[0] in ["Path", "Series"]:
            if ("-time" in args) or ("-values" in args):
                time, values = None, None
                if "-time" in args:
                    idx = args.index("-time")
                    time = list(args[idx + 1].split())
                    time = [float(i) for i in time]
                    args.pop(idx)
                    args.pop(idx)
                if "-values" in args:
                    idx = args.index("-values")
                    values = list(args[idx + 1].split())
                    values = [float(i) for i in values]
                    args.pop(idx)
                    args.pop(idx)
                if time and values:
                    args = [*args[:2], "-time", *time, "-values", *values, *args[2:]]
                elif values is None:
                    args = [*args[:2], "-time", *time, *args[2:]]
                else:
                    args = [*args[:2], "-values", *values, *args[2:]]
                txt = f"{self.prefix}timeSeries('Path', {args[1]}, *{args[2:]}){comments}"
                self.contents.append(txt)
            else:
                self.contents.append(f"{self.prefix}timeSeries{tuple(args)}{comments}")
        else:
            self.contents.append(f"{self.prefix}timeSeries{tuple(args)}{comments}")

    def _pattern(self, *args):
        args, comments = _process_args(args, keep_comments=self.keep_comments)
        if args[0].lower() != "uniformexcitation":
            if args[0].lower() == "plain" and isinstance(args[2], str):
                rprint(
                    f"[bold #d20962]Warning[/bold #d20962]: OpenSeesPy not support a str "
                    f"[bold #0099e5]{args[2]}[/bold #0099e5] "
                    f"followed [bold #ff4c4c]plain[/bold #ff4c4c], "
                    f"and a new [bold #f47721]timeSeries[/bold #f47721] is created with tag "
                    f"[bold #34bf49]{args[1]}[/bold #34bf49], "
                    f"please check this [bold #34bf49]pattern tag={args[1]}[/bold #34bf49]!"
                )
                tsargs = list(args[2].split())
                if len(tsargs) == 1:
                    self.contents.append(f"{self.prefix}timeSeries('{tsargs[0]}', {args[1]})")
                else:
                    self.contents.append(f"{self.prefix}timeSeries('{tsargs[0]}', {args[1]}, *{tsargs[1:]})")
                args = list(args)
                args[2] = args[1]
                args = tuple(args)
                self.contents.append(f"{self.prefix}pattern{args[:-1]}{comments}")
            else:
                self.contents.append(f"{self.prefix}pattern{args[:-1]}{comments}")
            txt = args[-1]
            txt.replace("\\n", "")
            self.interp.eval(txt)
        else:
            self.contents.append(f"{self.prefix}pattern{args}{comments}")

    def _load(self, *args):
        src = _get_cmds(args, "load", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _eleLoad(self, *args):
        src = _get_cmds(args, "eleLoad", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sp(self, *args):
        src = _get_cmds(args, "sp", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _groundMotion(self, *args):
        src = _get_cmds(args, "groundMotion", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _imposedMotion(self, *args):
        src = _get_cmds(args, "imposedMotion", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _mass(self, *args):
        src = _get_cmds(args, "mass", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _frictionModel(self, *args):
        src = _get_cmds(args, "frictionModel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _geomTransf(self, *args):
        src = _get_cmds(args, "geomTransf", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _region(self, *args):
        src = _get_cmds(args, "region", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _rayleigh(self, *args):
        src = _get_cmds(args, "rayleigh", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _block2D(self, *args):
        args, comments = _process_args(args, keep_comments=self.keep_comments)
        txt = args[-1]
        txt = txt.replace("\n", "").replace("\t", " ")
        crds = txt.split()
        crds = [_type_convert(i) for i in crds]
        self.contents.append(f"crds = {crds}")
        if isinstance(args[-2], str):
            eleargs = args[-2].split()
            eleargs = [_type_convert(i) for i in eleargs]
            args = args[:-2] + eleargs
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args]
            args.append("*crds")
        else:
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args[:-1]]
            args.append("*crds")
        txt = f"{self.prefix}block2D(" + ", ".join(args) + f"){comments}"
        self.contents.append(txt)

    def _block3D(self, *args):
        args, comments = _process_args(args, keep_comments=self.keep_comments)
        txt = args[-1]
        txt = txt.replace("\n", "").replace("\t", " ")
        crds = txt.split()
        crds = [_type_convert(i) for i in crds]
        self.contents.append(f"crds = {crds}")
        if isinstance(args[-2], str):
            eleargs = args[-2].split()
            eleargs = [_type_convert(i) for i in eleargs]
            args = args[:-2] + eleargs
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args]
            args.append("*crds")
        else:
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args[:-1]]
            args.append("*crds")
        txt = f"{self.prefix}block3D(" + ", ".join(args) + f"){comments}"
        self.contents.append(txt)

    def _ShallowFoundationGen(self, *args):
        src = _get_cmds(args, "ShallowFoundationGen", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _constraints(self, *args):
        src = _get_cmds(args, "constraints", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _numberer(self, *args):
        src = _get_cmds(args, "numberer", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _system(self, *args):
        src = _get_cmds(args, "system", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _test(self, *args):
        src = _get_cmds(args, "test", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _algorithm(self, *args):
        src = _get_cmds(args, "algorithm", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _integrator(self, *args):
        src = _get_cmds(args, "integrator", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _analysis(self, *args):
        src = _get_cmds(args, "analysis", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _eigen(self, *args):
        src = _get_cmds(args, "eigen", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)
        return None

    def _analyze(self, *args):
        src = _get_cmds(args, "analyze", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)
        return None

    def _modalProperties(self, *args):
        src = _get_cmds(args, "modalProperties", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)
        return None

    def _responseSpectrumAnalysis(self, *args):
        src = _get_cmds(args, "responseSpectrumAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _recorder(self, *args):
        src = _get_cmds(args, "recorder", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _record(self, *args):
        src = _get_cmds(args, "record", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _print(self, *args):
        src = _get_cmds(args, "printModel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _printA(self, *args):
        src = _get_cmds(args, "printA", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _logFile(self, *args):
        src = _get_cmds(args, "logFile", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _remove(self, *args):
        src = _get_cmds(args, "remove", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _loadConst(self, *args):
        src = _get_cmds(args, "loadConst", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _wipeAnalysis(self, *args):
        src = _get_cmds(args, "wipeAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _modalDamping(self, *args):
        src = _get_cmds(args, "modalDamping", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _database(self, *args):
        src = _get_cmds(args, "database", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getTime(self, *args):
        src = _get_cmds(args, "getTime", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setTime(self, *args):
        src = _get_cmds(args, "setTime", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _testUniaxialMaterial(self, *args):
        src = _get_cmds(args, "testUniaxialMaterial", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setStrain(self, *args):
        src = _get_cmds(args, "setStrain", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getStrain(self, *args):
        src = _get_cmds(args, "getStrain", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getStress(self, *args):
        src = _get_cmds(args, "getStress", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getTangent(self, *args):
        src = _get_cmds(args, "getTangent", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getDampTangent(self, *args):
        src = _get_cmds(args, "getDampTangent", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _reactions(self, *args):
        src = _get_cmds(args, "reactions", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeReaction(self, *args):
        src = _get_cmds(args, "nodeReaction", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeEigenvector(self, *args):
        src = _get_cmds(args, "nodeEigenvector", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setCreep(self, *args):
        src = _get_cmds(args, "setCreep", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _eleResponse(self, *args):
        src = _get_cmds(args, "eleResponse", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _reset(self, *args):
        src = _get_cmds(args, "reset", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _initialize(self, *args):
        src = _get_cmds(args, "initialize", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getLoadFactor(self, *args):
        src = _get_cmds(args, "getLoadFactor", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _build(self, *args):
        src = _get_cmds(args, "build", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _printGID(self, *args):
        src = _get_cmds(args, "printGID", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getCTestNorms(self, *args):
        src = _get_cmds(args, "getCTestNorms", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getCTestIter(self, *args):
        src = _get_cmds(args, "getCTestIter", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _save(self, *args):
        src = _get_cmds(args, "save", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _restore(self, *args):
        src = _get_cmds(args, "restore", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _eleForce(self, *args):
        src = _get_cmds(args, "eleForce", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _eleDynamicalForce(self, *args):
        src = _get_cmds(args, "eleDynamicalForce", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeUnbalance(self, *args):
        src = _get_cmds(args, "nodeUnbalance", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeDisp(self, *args):
        src = _get_cmds(args, "nodeDisp", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setNodeDisp(self, *args):
        src = _get_cmds(args, "setNodeDisp", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeVel(self, *args):
        src = _get_cmds(args, "nodeVel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setNodeVel(self, *args):
        src = _get_cmds(args, "setNodeVel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeAccel(self, *args):
        src = _get_cmds(args, "nodeAccel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setNodeAccel(self, *args):
        src = _get_cmds(args, "setNodeAccel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeResponse(self, *args):
        src = _get_cmds(args, "nodeResponse", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeCoord(self, *args):
        src = _get_cmds(args, "nodeCoord", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setNodeCoord(self, *args):
        src = _get_cmds(args, "setNodeCoord", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _updateElementDomain(self, *args):
        src = _get_cmds(args, "updateElementDomain", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNDMM(self, *args):
        src = _get_cmds(args, "getNDM", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNDFF(self, *args):
        src = _get_cmds(args, "getNDF", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _eleNodes(self, *args):
        src = _get_cmds(args, "eleNodes", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _eleType(self, *args):
        src = _get_cmds(args, "eleType", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeDOFs(self, *args):
        src = _get_cmds(args, "nodeDOFs", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeMass(self, *args):
        src = _get_cmds(args, "nodeMass", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodePressure(self, *args):
        src = _get_cmds(args, "nodePressure", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setNodePressure(self, *args):
        src = _get_cmds(args, "setNodePressure", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _nodeBounds(self, *args):
        src = _get_cmds(args, "nodeBounds", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _startTimer(self, *args):
        src = _get_cmds(args, "start", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _stopTimer(self, *args):
        src = _get_cmds(args, "stop", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _modalDampingQ(self, *args):
        src = _get_cmds(args, "modalDampingQ", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setElementRayleighDampingFactors(self, *args):
        src = _get_cmds(args, "setElementRayleighDampingFactors", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setPrecision(self, *args):
        src = _get_cmds(args, "setPrecision", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _searchPeerNGA(self, *args):
        src = _get_cmds(args, "searchPeerNGA", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _domainChange(self, *args):
        src = _get_cmds(args, "domainChange", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _defaultUnits(self, *args):
        src = _get_cmds(args, "defaultUnits", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _stripXML(self, *args):
        src = _get_cmds(args, "stripXML", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _convertBinaryToText(self, *args):
        src = _get_cmds(args, "convertBinaryToText", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _convertTextToBinary(self, *args):
        src = _get_cmds(args, "convertTextToBinary", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getEleTags(self, *args):
        src = _get_cmds(args, "getEleTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getCrdTransfTags(self, *args):
        src = _get_cmds(args, "getCrdTransfTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNodeTags(self, *args):
        src = _get_cmds(args, "getNodeTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getParamTags(self, *args):
        src = _get_cmds(args, "getParamTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getParamValue(self, *args):
        src = _get_cmds(args, "getParamValue", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionForce(self, *args):
        src = _get_cmds(args, "sectionForce", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionDeformation(self, *args):
        src = _get_cmds(args, "sectionDeformation", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionStiffness(self, *args):
        src = _get_cmds(args, "sectionStiffness", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionFlexibility(self, *args):
        src = _get_cmds(args, "sectionFlexibility", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionLocation(self, *args):
        src = _get_cmds(args, "sectionLocation", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionWeight(self, *args):
        src = _get_cmds(args, "sectionWeight", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionTag(self, *args):
        src = _get_cmds(args, "sectionTag", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sectionDisplacement(self, *args):
        src = _get_cmds(args, "sectionDisplacement", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _cbdiDisplacement(self, *args):
        src = _get_cmds(args, "cbdiDisplacement", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _basicDeformation(self, *args):
        src = _get_cmds(args, "basicDeformation", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _basicForce(self, *args):
        src = _get_cmds(args, "basicForce", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _basicStiffness(self, *args):
        src = _get_cmds(args, "basicStiffness", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _InitialStateAnalysis(self, *args):
        src = _get_cmds(args, "InitialStateAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _totalCPU(self, *args):
        src = _get_cmds(args, "totalCPU", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _solveCPU(self, *args):
        src = _get_cmds(args, "solveCPU", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _accelCPU(self, *args):
        src = _get_cmds(args, "accelCPU", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _numFact(self, *args):
        src = _get_cmds(args, "numFact", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _numIter(self, *args):
        src = _get_cmds(args, "numIter", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _systemSize(self, *args):
        src = _get_cmds(args, "systemSize", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _version(self, *args):
        src = _get_cmds(args, "version", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setMaxOpenFiles(self, *args):
        src = _get_cmds(args, "setMaxOpenFiles", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _limitCurve(self, *args):
        src = _get_cmds(args, "limitCurve", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setElementRayleighFactors(self, *args):
        src = _get_cmds(args, "setElementRayleighFactors", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _mesh(self, *args):
        src = _get_cmds(args, "mesh", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _remesh(self, *args):
        src = _get_cmds(args, "remesh", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _parameter(self, *args):
        src = _get_cmds(args, "parameter", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _addToParameter(self, *args):
        src = _get_cmds(args, "addToParameter", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _updateParameter(self, *args):
        src = _get_cmds(args, "updateParameter", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setParameter(self, *args):
        src = _get_cmds(args, "setParameter", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getPID(self, *args):
        src = _get_cmds(args, "getPID", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNP(self, *args):
        src = _get_cmds(args, "getNP", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _barrier(self, *args):
        src = _get_cmds(args, "barrier", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _send(self, *args):
        src = _get_cmds(args, "send", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _recv(self, *args):
        src = _get_cmds(args, "recv", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _Bcast(self, *args):
        src = _get_cmds(args, "Bcast", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _computeGradients(self, *args):
        src = _get_cmds(args, "computeGradients", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sensitivityAlgorithm(self, *args):
        src = _get_cmds(args, "sensitivityAlgorithm", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sensNodeDisp(self, *args):
        src = _get_cmds(args, "sensNodeDisp", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sensNodeVel(self, *args):
        src = _get_cmds(args, "sensNodeVel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sensNodeAccel(self, *args):
        src = _get_cmds(args, "sensNodeAccel", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sensLambda(self, *args):
        src = _get_cmds(args, "sensLambda", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sensSectionForce(self, *args):
        src = _get_cmds(args, "sensSectionForce", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sensNodePressure(self, *args):
        src = _get_cmds(args, "sensNodePressure", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNumElements(self, *args):
        src = _get_cmds(args, "getNumElements", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getEleClassTags(self, *args):
        src = _get_cmds(args, "getEleClassTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getEleLoadClassTags(self, *args):
        src = _get_cmds(args, "getEleLoadClassTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getEleLoadTags(self, *args):
        src = _get_cmds(args, "getEleLoadTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getEleLoadData(self, *args):
        src = _get_cmds(args, "getEleLoadData", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNodeLoadTags(self, *args):
        src = _get_cmds(args, "getNodeLoadTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNodeLoadData(self, *args):
        src = _get_cmds(args, "getNodeLoadData", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _randomVariable(self, *args):
        src = _get_cmds(args, "randomVariable", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getRVTags(self, *args):
        src = _get_cmds(args, "getRVTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getRVMean(self, *args):
        src = _get_cmds(args, "getRVMean", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getRVStdv(self, *args):
        src = _get_cmds(args, "getRVStdv", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getRVPDF(self, *args):
        src = _get_cmds(args, "getRVPDF", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getRVCDF(self, *args):
        src = _get_cmds(args, "getRVCDF", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getRVInverseCDF(self, *args):
        src = _get_cmds(args, "getRVInverseCDF", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _addCorrelate(self, *args):
        src = _get_cmds(args, "addCorrelate", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _correlate(self, *args):
        src = _get_cmds(args, "correlate", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _performanceFunction(self, *args):
        src = _get_cmds(args, "performanceFunction", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _gradPerformanceFunction(self, *args):
        src = _get_cmds(args, "gradPerformanceFunction", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _transformUtoX(self, *args):
        src = _get_cmds(args, "transformUtoX", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _wipeReliability(self, *args):
        src = _get_cmds(args, "wipeReliability", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _updateMaterialStage(self, *args):
        src = _get_cmds(args, "updateMaterialStage", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _sdfResponse(self, *args):
        src = _get_cmds(args, "sdfResponse", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _probabilityTransformation(self, *args):
        src = _get_cmds(args, "probabilityTransformation", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _startPoint(self, *args):
        src = _get_cmds(args, "startPoint", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _randomNumberGenerator(self, *args):
        src = _get_cmds(args, "randomNumberGenerator", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _reliabilityConvergenceCheck(self, *args):
        src = _get_cmds(args, "reliabilityConvergenceCheck", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _searchDirection(self, *args):
        src = _get_cmds(args, "searchDirection", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _meritFunctionCheck(self, *args):
        src = _get_cmds(args, "meritFunctionCheck", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _stepSizeRule(self, *args):
        src = _get_cmds(args, "stepSizeRule", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _rootFinding(self, *args):
        src = _get_cmds(args, "rootFinding", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _functionEvaluator(self, *args):
        src = _get_cmds(args, "functionEvaluator", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _gradientEvaluator(self, *args):
        src = _get_cmds(args, "gradientEvaluator", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _runFOSMAnalysis(self, *args):
        src = _get_cmds(args, "runFOSMAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _findDesignPoint(self, *args):
        src = _get_cmds(args, "findDesignPoint", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _runFORMAnalysis(self, *args):
        src = _get_cmds(args, "runFORMAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getLSFTags(self, *args):
        src = _get_cmds(args, "getLSFTags", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _runImportanceSamplingAnalysis(self, *args):
        src = _get_cmds(args, "runImportanceSamplingAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _IGA(self, *args):
        src = _get_cmds(args, "IGA", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _NDTest(self, *args):
        src = _get_cmds(args, "NDTest", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _getNumThreads(self, *args):
        src = _get_cmds(args, "getNumThreads", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setNumThreads(self, *args):
        src = _get_cmds(args, "setNumThreads", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _setStartNodeTag(self, *args):
        src = _get_cmds(args, "setStartNodeTag", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _hystereticBackbone(self, *args):
        src = _get_cmds(args, "hystereticBackbone", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _stiffnessDegradation(self, *args):
        src = _get_cmds(args, "stiffnessDegradation", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _strengthDegradation(self, *args):
        src = _get_cmds(args, "strengthDegradation", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _unloadingRule(self, *args):
        src = _get_cmds(args, "unloadingRule", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _partition(self, *args):
        src = _get_cmds(args, "partition", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _pc(self, *args):
        src = _get_cmds(args, "pressureConstraint", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    def _domainCommitTag(self, *args):
        src = _get_cmds(args, "domainCommitTag", prefix=self.prefix, keep_comments=self.keep_comments)
        self.contents.append(src)

    @staticmethod
    def _display(*args):
        print(f"This <display {args}> function will be ignored!")

    @staticmethod
    def _prp(*args):
        print(f"This display <prp {args}> function will be ignored!")

    @staticmethod
    def _vup(*args):
        print(f"This display <vup {args}> function will be ignored!")

    @staticmethod
    def _vpn(*args):
        print(f"This display <vpn {args}> function will be ignored!")

    @staticmethod
    def _vrp(*args):
        print(f"This display <vrp {args}> function will be ignored!")

    def _createcommand(self):
        self.interp.createcommand("comments___", self._comments)
        self.interp.createcommand("puts", self._puts)
        self.interp.createcommand("wipe", self._wipe)
        self.interp.createcommand("model", self._model)
        self.interp.createcommand("node", self._node)
        self.interp.createcommand("fix", self._fix)
        self.interp.createcommand("fixX", self._fixX)
        self.interp.createcommand("fixY", self._fixY)
        self.interp.createcommand("fixZ", self._fixZ)
        self.interp.createcommand("equalDOF", self._equalDOF)
        self.interp.createcommand("equalDOF_Mixed", self._equalDOF_Mixed)
        self.interp.createcommand("rigidDiaphragm", self._rigidDiaphragm)
        self.interp.createcommand("rigidLink", self._rigidLink)
        self.interp.createcommand("element", self._element)
        self.interp.createcommand("timeSeries", self._timeSeries)
        # self.interp.createcommand('Series', _timeSeries)
        self.interp.createcommand("pattern", self._pattern)
        self.interp.createcommand("load", self._load)
        self.interp.createcommand("eleLoad", self._eleLoad)
        self.interp.createcommand("sp", self._sp)
        self.interp.createcommand("groundMotion", self._groundMotion)
        self.interp.createcommand("imposedMotion", self._imposedMotion)
        self.interp.createcommand("imposedSupportMotion", self._imposedMotion)
        self.interp.createcommand("mass", self._mass)
        self.interp.createcommand("uniaxialMaterial", self._uniaxialMaterial)
        self.interp.createcommand("nDMaterial", self._nDMaterial)
        self.interp.createcommand("beamIntegration", self._beamIntegration)
        self.interp.createcommand("section", self._section)
        self.interp.createcommand("fiber", self._fiber)
        self.interp.createcommand("patch", self._patch)
        self.interp.createcommand("layer", self._layer)
        self.interp.createcommand("frictionModel", self._frictionModel)
        self.interp.createcommand("geomTransf", self._geomTransf)
        self.interp.createcommand("region", self._region)
        self.interp.createcommand("rayleigh", self._rayleigh)
        self.interp.createcommand("block2D", self._block2D)
        self.interp.createcommand("block2d", self._block2D)
        self.interp.createcommand("block3D", self._block3D)
        self.interp.createcommand("block3d", self._block3D)
        self.interp.createcommand("ShallowFoundationGen", self._ShallowFoundationGen)
        self.interp.createcommand("constraints", self._constraints)
        self.interp.createcommand("numberer", self._numberer)
        self.interp.createcommand("system", self._system)
        self.interp.createcommand("test", self._test)
        self.interp.createcommand("algorithm", self._algorithm)
        self.interp.createcommand("integrator", self._integrator)
        self.interp.createcommand("analysis", self._analysis)
        self.interp.createcommand("eigen", self._eigen)
        self.interp.createcommand("analyze", self._analyze)
        self.interp.createcommand("modalProperties", self._modalProperties)
        self.interp.createcommand("responseSpectrumAnalysis", self._responseSpectrumAnalysis)
        self.interp.createcommand("record", self._record)
        self.interp.createcommand("recorder", self._recorder)
        self.interp.createcommand("print", self._print)
        self.interp.createcommand("printA", self._printA)
        self.interp.createcommand("logFile", self._logFile)
        self.interp.createcommand("remove", self._remove)
        self.interp.createcommand("loadConst", self._loadConst)
        self.interp.createcommand("wipeAnalysis", self._wipeAnalysis)
        self.interp.createcommand("modalDamping", self._modalDamping)
        self.interp.createcommand("database", self._database)
        self.interp.createcommand("getTime", self._getTime)
        self.interp.createcommand("setTime", self._setTime)
        self.interp.createcommand("testUniaxialMaterial", self._testUniaxialMaterial)
        self.interp.createcommand("setStrain", self._setStrain)
        self.interp.createcommand("getStrain", self._getStrain)
        self.interp.createcommand("getStress", self._getStress)
        self.interp.createcommand("getTangent", self._getTangent)
        self.interp.createcommand("getDampTangent", self._getDampTangent)
        self.interp.createcommand("reactions", self._reactions)
        self.interp.createcommand("nodeReaction", self._nodeReaction)
        self.interp.createcommand("remove", self._remove)
        self.interp.createcommand("nodeEigenvector", self._nodeEigenvector)
        self.interp.createcommand("setCreep", self._setCreep)
        self.interp.createcommand("eleResponse", self._eleResponse)
        self.interp.createcommand("reset", self._reset)
        self.interp.createcommand("initialize", self._initialize)
        self.interp.createcommand("getLoadFactor", self._getLoadFactor)
        self.interp.createcommand("build", self._build)
        self.interp.createcommand("printGID", self._printGID)
        self.interp.createcommand("testNorm", self._getCTestNorms)
        self.interp.createcommand("testIter", self._getCTestIter)
        self.interp.createcommand("save", self._save)
        self.interp.createcommand("restore", self._restore)
        self.interp.createcommand("eleForce", self._eleForce)
        self.interp.createcommand("eleDynamicalForce", self._eleDynamicalForce)
        self.interp.createcommand("nodeUnbalance", self._nodeUnbalance)
        self.interp.createcommand("nodeDisp", self._nodeDisp)
        self.interp.createcommand("setNodeDisp", self._setNodeDisp)
        self.interp.createcommand("nodeVel", self._nodeVel)
        self.interp.createcommand("setNodeVel", self._setNodeVel)
        self.interp.createcommand("nodeAccel", self._nodeAccel)
        self.interp.createcommand("setNodeAccel", self._setNodeAccel)
        self.interp.createcommand("nodeResponse", self._nodeResponse)
        self.interp.createcommand("nodeCoord", self._nodeCoord)
        self.interp.createcommand("setNodeCoord", self._setNodeCoord)
        self.interp.createcommand("updateElementDomain", self._updateElementDomain)
        self.interp.createcommand("getNDM", self._getNDMM)
        self.interp.createcommand("getNDF", self._getNDFF)
        self.interp.createcommand("eleNodes", self._eleNodes)
        self.interp.createcommand("eleType", self._eleType)
        self.interp.createcommand("nodeDOFs", self._nodeDOFs)
        self.interp.createcommand("nodeMass", self._nodeMass)
        self.interp.createcommand("nodePressure", self._nodePressure)
        self.interp.createcommand("setNodePressure", self._setNodePressure)
        self.interp.createcommand("nodeBounds", self._nodeBounds)
        self.interp.createcommand("start", self._startTimer)
        self.interp.createcommand("stop", self._stopTimer)
        self.interp.createcommand("modalDampingQ", self._modalDampingQ)
        self.interp.createcommand("setElementRayleighDampingFactors", self._setElementRayleighDampingFactors)
        self.interp.createcommand("setPrecision", self._setPrecision)
        self.interp.createcommand("searchPeerNGA", self._searchPeerNGA)
        self.interp.createcommand("domainChange", self._domainChange)
        self.interp.createcommand("defaultUnits", self._defaultUnits)
        self.interp.createcommand("stripXML", self._stripXML)
        self.interp.createcommand("convertBinaryToText", self._convertBinaryToText)
        self.interp.createcommand("convertTextToBinary", self._convertTextToBinary)
        self.interp.createcommand("getEleTags", self._getEleTags)
        self.interp.createcommand("getCrdTransfTags", self._getCrdTransfTags)
        self.interp.createcommand("getNodeTags", self._getNodeTags)
        self.interp.createcommand("getParamTags", self._getParamTags)
        self.interp.createcommand("getParamValue", self._getParamValue)
        self.interp.createcommand("sectionForce", self._sectionForce)
        self.interp.createcommand("sectionDeformation", self._sectionDeformation)
        self.interp.createcommand("sectionStiffness", self._sectionStiffness)
        self.interp.createcommand("sectionFlexibility", self._sectionFlexibility)
        self.interp.createcommand("sectionLocation", self._sectionLocation)
        self.interp.createcommand("sectionWeight", self._sectionWeight)
        self.interp.createcommand("sectionTag", self._sectionTag)
        self.interp.createcommand("sectionDisplacement", self._sectionDisplacement)
        self.interp.createcommand("cbdiDisplacement", self._cbdiDisplacement)
        self.interp.createcommand("basicDeformation", self._basicDeformation)
        self.interp.createcommand("basicForce", self._basicForce)
        self.interp.createcommand("basicStiffness", self._basicStiffness)
        self.interp.createcommand("InitialStateAnalysis", self._InitialStateAnalysis)
        self.interp.createcommand("totalCPU", self._totalCPU)
        self.interp.createcommand("solveCPU", self._solveCPU)
        self.interp.createcommand("accelCPU", self._accelCPU)
        self.interp.createcommand("numFact", self._numFact)
        self.interp.createcommand("numIter", self._numIter)
        self.interp.createcommand("systemSize", self._systemSize)
        self.interp.createcommand("version", self._version)
        self.interp.createcommand("setMaxOpenFiles", self._setMaxOpenFiles)
        self.interp.createcommand("limitCurve", self._limitCurve)

        self.interp.createcommand("equalDOF_Mixed", self._equalDOF_Mixed)
        self.interp.createcommand("setElementRayleighFactors", self._setElementRayleighFactors)
        self.interp.createcommand("mesh", self._mesh)
        self.interp.createcommand("remesh", self._remesh)
        self.interp.createcommand("parameter", self._parameter)
        self.interp.createcommand("addToParameter", self._addToParameter)
        self.interp.createcommand("updateParameter", self._updateParameter)
        self.interp.createcommand("setParameter", self._setParameter)
        self.interp.createcommand("getPID", self._getPID)
        self.interp.createcommand("getNP", self._getNP)
        self.interp.createcommand("barrier", self._barrier)
        self.interp.createcommand("send", self._send)
        self.interp.createcommand("recv", self._recv)
        self.interp.createcommand("Bcast", self._Bcast)
        self.interp.createcommand("computeGradients", self._computeGradients)
        self.interp.createcommand("sensitivityAlgorithm", self._sensitivityAlgorithm)
        self.interp.createcommand("sensNodeDisp", self._sensNodeDisp)
        self.interp.createcommand("sensNodeVel", self._sensNodeVel)
        self.interp.createcommand("sensNodeAccel", self._sensNodeAccel)
        self.interp.createcommand("sensLambda", self._sensLambda)
        self.interp.createcommand("sensSectionForce", self._sensSectionForce)
        self.interp.createcommand("sensNodePressure", self._sensNodePressure)
        self.interp.createcommand("getNumElements", self._getNumElements)
        self.interp.createcommand("getEleClassTags", self._getEleClassTags)
        self.interp.createcommand("getEleLoadClassTags", self._getEleLoadClassTags)
        self.interp.createcommand("getEleLoadTags", self._getEleLoadTags)
        self.interp.createcommand("getEleLoadData", self._getEleLoadData)
        self.interp.createcommand("getNodeLoadTags", self._getNodeLoadTags)
        self.interp.createcommand("getNodeLoadData", self._getNodeLoadData)
        self.interp.createcommand("randomVariable", self._randomVariable)
        self.interp.createcommand("getRVTags", self._getRVTags)
        self.interp.createcommand("getMean", self._getRVMean)
        self.interp.createcommand("getStdv", self._getRVStdv)
        self.interp.createcommand("getPDF", self._getRVPDF)
        self.interp.createcommand("getCDF", self._getRVCDF)
        self.interp.createcommand("getInverseCDF", self._getRVInverseCDF)
        self.interp.createcommand("correlate", self._correlate)
        self.interp.createcommand("performanceFunction", self._performanceFunction)
        self.interp.createcommand("gradPerformanceFunction", self._gradPerformanceFunction)
        self.interp.createcommand("transformUtoX", self._transformUtoX)
        self.interp.createcommand("wipeReliability", self._wipeReliability)
        self.interp.createcommand("updateMaterialStage", self._updateMaterialStage)
        self.interp.createcommand("sdfResponse", self._sdfResponse)
        self.interp.createcommand("probabilityTransformation", self._probabilityTransformation)
        self.interp.createcommand("startPoint", self._startPoint)
        self.interp.createcommand("randomNumberGenerator", self._randomNumberGenerator)
        self.interp.createcommand("reliabilityConvergenceCheck", self._reliabilityConvergenceCheck)
        self.interp.createcommand("searchDirection", self._searchDirection)
        self.interp.createcommand("meritFunctionCheck", self._meritFunctionCheck)
        self.interp.createcommand("stepSizeRule", self._stepSizeRule)
        self.interp.createcommand("rootFinding", self._rootFinding)
        self.interp.createcommand("functionEvaluator", self._functionEvaluator)
        self.interp.createcommand("gradientEvaluator", self._gradientEvaluator)
        self.interp.createcommand("runFOSMAnalysis", self._runFOSMAnalysis)
        self.interp.createcommand("findDesignPoint", self._findDesignPoint)
        self.interp.createcommand("runFORMAnalysis", self._runFORMAnalysis)
        self.interp.createcommand("getLSFTags", self._getLSFTags)
        self.interp.createcommand("runImportanceSamplingAnalysis", self._runImportanceSamplingAnalysis)
        self.interp.createcommand("IGA", self._IGA)
        self.interp.createcommand("NDTest", self._NDTest)
        self.interp.createcommand("getNumThreads", self._getNumThreads)
        self.interp.createcommand("setNumThreads", self._setNumThreads)
        self.interp.createcommand("setStartNodeTag", self._setStartNodeTag)
        self.interp.createcommand("hystereticBackbone", self._hystereticBackbone)
        self.interp.createcommand("stiffnessDegradation", self._stiffnessDegradation)
        self.interp.createcommand("strengthDegradation", self._strengthDegradation)
        self.interp.createcommand("unloadingRule", self._unloadingRule)
        self.interp.createcommand("partition", self._partition)
        self.interp.createcommand("pressureConstraint", self._pc)
        self.interp.createcommand("domainCommitTag", self._domainCommitTag)
        # ------------------------------------------------------------
        self.interp.createcommand("display", self._display)
        self.interp.createcommand("prp", self._prp)
        self.interp.createcommand("vup", self._vup)
        self.interp.createcommand("vpn", self._vpn)
        self.interp.createcommand("vrp", self._vrp)

    def eval(self, contents):
        self._createcommand()
        self.interp.eval(contents)

    def get_interp(self):
        return self.interp

    def get_opspy_cmds(self):
        return self.contents
