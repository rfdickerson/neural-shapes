#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string inputPath;
    std::string outputBinPath;
    std::string outputJsonPath;
    std::string gridName;
    int dim = 128;
    bool clamp01 = true;
};

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0
        << " --input <cloud.vdb> --output-bin <volume.bin> --output-json <volume.json>"
        << " [--grid <name>] [--dim <N>] [--no-clamp]\n";
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
        } else if (arg == "--dim") {
            std::string value;
            if (!consume(value)) return false;
            options.dim = std::atoi(value.c_str());
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
    if (options.dim < 4 || options.dim > 1024) {
        std::cerr << "--dim must be in [4, 1024], got " << options.dim << "\n";
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

        const int dim = options.dim;
        std::vector<float> data;
        data.resize(static_cast<size_t>(dim) * static_cast<size_t>(dim) * static_cast<size_t>(dim), 0.0f);

        size_t idx = 0;
        for (int z = 0; z < dim; ++z) {
            const double tz = (static_cast<double>(z) + 0.5) / static_cast<double>(dim);
            const double wz = worldMin.z() + (worldMax.z() - worldMin.z()) * tz;
            for (int y = 0; y < dim; ++y) {
                const double ty = (static_cast<double>(y) + 0.5) / static_cast<double>(dim);
                const double wy = worldMin.y() + (worldMax.y() - worldMin.y()) * ty;
                for (int x = 0; x < dim; ++x) {
                    const double tx = (static_cast<double>(x) + 0.5) / static_cast<double>(dim);
                    const double wx = worldMin.x() + (worldMax.x() - worldMin.x()) * tx;
                    float value = sampler.wsSample(openvdb::Vec3d(wx, wy, wz));
                    if (options.clamp01) {
                        if (value < 0.0f) value = 0.0f;
                        if (value > 1.0f) value = 1.0f;
                    }
                    data[idx++] = value;
                }
            }
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

        {
            std::ofstream out(options.outputJsonPath);
            if (!out) {
                throw std::runtime_error("Failed to open output json file: " + options.outputJsonPath);
            }
            out << "{\n";
            out << "  \"format\": \"dense_f32\",\n";
            out << "  \"dims\": [" << dim << ", " << dim << ", " << dim << "],\n";
            out << "  \"domain\": \"normalized_cube\",\n";
            out << "  \"domain_min\": [-1.0, -1.0, -1.0],\n";
            out << "  \"domain_max\": [1.0, 1.0, 1.0],\n";
            out << "  \"source\": {\n";
            out << "    \"vdb_path\": \"" << options.inputPath << "\",\n";
            out << "    \"grid\": \"" << actualGridName << "\",\n";
            out << "    \"active_world_min\": " << vecToJson(worldMin) << ",\n";
            out << "    \"active_world_max\": " << vecToJson(worldMax) << "\n";
            out << "  }\n";
            out << "}\n";
        }

        float minValue = std::numeric_limits<float>::max();
        float maxValue = std::numeric_limits<float>::lowest();
        double sum = 0.0;
        for (const float value : data) {
            if (value < minValue) minValue = value;
            if (value > maxValue) maxValue = value;
            sum += value;
        }
        const double mean = sum / static_cast<double>(data.size());

        std::cout << "Converted VDB to dense volume:\n";
        std::cout << "  input:  " << options.inputPath << "\n";
        std::cout << "  grid:   " << actualGridName << "\n";
        std::cout << "  dims:   " << dim << " x " << dim << " x " << dim << "\n";
        std::cout << "  output: " << options.outputBinPath << "\n";
        std::cout << "  meta:   " << options.outputJsonPath << "\n";
        std::cout << "  stats:  min=" << minValue << " max=" << maxValue << " mean=" << mean << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
