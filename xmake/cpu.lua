target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    -- For AVX512
    if is_plat("linux")and is_arch("x86_64") then
        add_cxflags("-mavx512f -mfma -mf16c")
        add_mxflags("-mavx512f -mfma -mf16c")
    end
    
    -- For AVX2
    -- if is_plat("linux")and is_arch("x86_64") then
    --     add_cxflags("-mavx2 -mfma -mf16c")
    --     add_mxflags("-mavx2 -mfma -mf16c")
    -- end

    add_packages("openmp", "openblas")

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

