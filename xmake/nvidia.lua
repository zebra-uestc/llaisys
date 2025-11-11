target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart")

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cxxflags("/FS")
    else
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxxflags("-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()


target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart")
    add_cugencodes("native")

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cuflags("-Xcompiler=/W3", "-Xcompiler=/WX")
        add_cxxflags("/FS")
    else
        add_cuflags("-Xcompiler=-Wall", "-Xcompiler=-Werror")
        add_cuflags("-Xcompiler=-fPIC")
        add_cuflags("--extended-lambda")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxxflags("-fPIC")
        add_cuflags("--expt-relaxed-constexpr")
    end

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations")

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
