#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string inputPath;
    std::string outputBinPath;
    std::string outputJsonPath;
    std::string gridName;
    std::uint64_t sampleCount = 4'000'000;
    std::uint32_t seed = 1337u;
    bool clamp01 = true;
};

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0
        << " --input <cloud.vdb> --output-bin <samples.bin> --output-json <samples.json>"
        << " [--grid <name>] [--samples <count>] [--seed <u32>] [--no-clamp]\n";
}

bool parseArgs(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto consume = [&](std::string& out) -> bool {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };

        if (arg == "--input") {
            if (!consume(options.inputPath)) return false;
        } else if (arg == "--output-bin") {
            if (!consume(options.outputBinPath)) return false;
        } else if (arg == "--output-json") {
            if (!consume(options.outputJsonPath)) return false;
        } else if (arg == "--grid") {
            if (!consume(options.gridName)) return false;
        } else if (arg == "--samples") {
            std::string value;
            if (!consume(value)) return false;
            options.sampleCount = static_cast<std::uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
        } else if (arg == "--seed") {
            std::string value;
            if (!consume(value)) return false;
            options.seed = static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (arg == "--no-clamp") {
            options.clamp01 = false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (options.inputPath.empty() || options.outputBinPath.empty() || options.outputJsonPath.empty()) {
        return false;
    }
    if (options.sampleCount == 0 || options.sampleCount > 100'000'000ull) {
        std::cerr << "--samples must be in [1, 100000000], got " << options.sampleCount << "\n";
        return false;
    }
    return true;
}

openvdb::FloatGrid::Ptr loadFloatGrid(openvdb::io::File& file, const std::string& requestedName)
{
    if (!requestedName.empty()) {
        openvdb::GridBase::Ptr base = file.readGrid(requestedName);
        if (!base) {
            throw std::runtime_error("Failed to read requested grid '" + requestedName + "'");
        }
        if (!base->isType<openvdb::FloatGrid>()) {
            throw std::runtime_error("Grid '" + requestedName + "' is not FloatGrid.");
        }
        return openvdb::gridPtrCast<openvdb::FloatGrid>(base);
    }

    for (openvdb::io::File::NameIterator n = file.beginName(); n != file.endName(); ++n) {
        openvdb::GridBase::Ptr base = file.readGrid(n.gridName());
        if (base && base->isType<openvdb::FloatGrid>()) {
            return openvdb::gridPtrCast<openvdb::FloatGrid>(base);
        }
    }
    throw std::runtime_error("No FloatGrid found in VDB file.");
}

std::string vecToJson(const openvdb::Vec3d& v)
{
    std::ostringstream ss;
    ss << std::setprecision(9) << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
    return ss.str();
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parseArgs(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        openvdb::initialize();
        openvdb::io::File file(options.inputPath);
        file.open();

        openvdb::FloatGrid::Ptr grid = loadFloatGrid(file, options.gridName);
        const std::string actualGridName = grid->getName();

        const openvdb::CoordBBox activeBBox = grid->evalActiveVoxelBoundingBox();
        if (activeBBox.empty()) {
            throw std::runtime_error("Selected grid has no active voxels.");
        }

        openvdb::Coord maxCoord = activeBBox.max();
        maxCoord += openvdb::Coord(1, 1, 1);

        const openvdb::Vec3d worldMin = grid->transform().indexToWorld(activeBBox.min().asVec3d());
        const openvdb::Vec3d worldMax = grid->transform().indexToWorld(maxCoord.asVec3d());

        openvdb::tools::GridSampler<openvdb::FloatTree, openvdb::tools::BoxSampler> sampler(
            grid->tree(), grid->transform());

        std::mt19937 rng(options.seed);
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        std::vector<float> data;
        data.resize(static_cast<size_t>(options.sampleCount) * 4u, 0.0f);

        float minValue = std::numeric_limits<float>::max();
        float maxValue = std::numeric_limits<float>::lowest();
        double sum = 0.0;

        for (std::uint64_t i = 0; i < options.sampleCount; ++i) {
            const double u = uniform01(rng);
            const double v = uniform01(rng);
            const double w = uniform01(rng);

            const double wx = worldMin.x() + (worldMax.x() - worldMin.x()) * u;
            const double wy = worldMin.y() + (worldMax.y() - worldMin.y()) * v;
            const double wz = worldMin.z() + (worldMax.z() - worldMin.z()) * w;

            float value = sampler.wsSample(openvdb::Vec3d(wx, wy, wz));
            if (options.clamp01) {
                value = std::clamp(value, 0.0f, 1.0f);
            }

            const size_t base = static_cast<size_t>(i) * 4u;
            data[base + 0] = static_cast<float>(u * 2.0 - 1.0);
            data[base + 1] = static_cast<float>(v * 2.0 - 1.0);
            data[base + 2] = static_cast<float>(w * 2.0 - 1.0);
            data[base + 3] = value;

            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
            sum += static_cast<double>(value);
        }

        file.close();

        {
            std::ofstream out(options.outputBinPath, std::ios::binary);
            if (!out) {
                throw std::runtime_error("Failed to open output bin file: " + options.outputBinPath);
            }
            out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
            if (!out.good()) {
                throw std::runtime_error("Failed writing output bin file: " + options.outputBinPath);
            }
        }

        const double mean = sum / static_cast<double>(options.sampleCount);
        {
            std::ofstream out(options.outputJsonPath);
            if (!out) {
                throw std::runtime_error("Failed to open output json file: " + options.outputJsonPath);
            }
            out << "{\n";
            out << "  \"format\": \"xyz_density_f32\",\n";
            out << "  \"count\": " << options.sampleCount << ",\n";
            out << "  \"stride_floats\": 4,\n";
            out << "  \"domain\": \"normalized_cube\",\n";
            out << "  \"domain_min\": [-1.0, -1.0, -1.0],\n";
            out << "  \"domain_max\": [1.0, 1.0, 1.0],\n";
            out << "  \"source\": {\n";
            out << "    \"vdb_path\": \"" << options.inputPath << "\",\n";
            out << "    \"grid\": \"" << actualGridName << "\",\n";
            out << "    \"active_world_min\": " << vecToJson(worldMin) << ",\n";
            out << "    \"active_world_max\": " << vecToJson(worldMax) << "\n";
            out << "  },\n";
            out << "  \"stats\": {\n";
            out << "    \"density_min\": " << minValue << ",\n";
            out << "    \"density_max\": " << maxValue << ",\n";
            out << "    \"density_mean\": " << mean << "\n";
            out << "  }\n";
            out << "}\n";
        }

        std::cout << "Generated sparse VDB samples:\n";
        std::cout << "  input:   " << options.inputPath << "\n";
        std::cout << "  grid:    " << actualGridName << "\n";
        std::cout << "  samples: " << options.sampleCount << "\n";
        std::cout << "  output:  " << options.outputBinPath << "\n";
        std::cout << "  meta:    " << options.outputJsonPath << "\n";
        std::cout << "  stats:   min=" << minValue << " max=" << maxValue << " mean=" << mean << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
