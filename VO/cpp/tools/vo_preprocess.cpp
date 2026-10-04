// vo_preprocess - dump images as the C++ runtime sees them (parity tests).
//
//   vo_preprocess decode  <image> <color 0|1> <out.raw>                  file -> Pillow RGB / L
//   vo_preprocess resize  <image> <color 0|1> <width> <height> <out.raw> ... -> Pillow BILINEAR resize
//   vo_preprocess runtime <onnx_dir> <image> <out.raw>                   the frontend's float32 (C, H, W)
//
// Output: one text line "<width> <height> <channels> <u8|f32>\n", then the
// raw pixels (HWC uint8, or CHW float32 for `runtime`).
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>

#include "vo_stream.hpp"

namespace {

void writeImage(const std::string& path, const vo::ImageU8& image) {
  std::ofstream out(path, std::ios::binary);
  out << image.width << ' ' << image.height << ' ' << image.channels << " u8\n";
  out.write(reinterpret_cast<const char*>(image.data.data()), static_cast<std::streamsize>(image.data.size()));
  if (!out) throw std::runtime_error("cannot write " + path);
}

int usage() {
  std::fprintf(stderr,
               "usage: vo_preprocess decode <image> <color 0|1> <out.raw>\n"
               "       vo_preprocess resize <image> <color 0|1> <width> <height> <out.raw>\n"
               "       vo_preprocess runtime <onnx_dir> <image> <out.raw>\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) return usage();
    const std::string mode = argv[1];
    if (mode == "decode" && argc == 5) {
      writeImage(argv[4], vo::loadImageFile(argv[2], std::atoi(argv[3]) != 0));
    } else if (mode == "resize" && argc == 7) {
      const vo::ImageU8 image = vo::loadImageFile(argv[2], std::atoi(argv[3]) != 0);
      writeImage(argv[6], vo::resizeBilinearPIL(image, std::atoi(argv[4]), std::atoi(argv[5])));
    } else if (mode == "runtime" && argc == 5) {
      vo::Options options;
      options.async_frontend = false;
      vo::StreamRuntime runtime(argv[2], options);
      const vo::Settings& s = runtime.settings();
      const vo::ImageU8 image = vo::loadImageFile(argv[3], s.color);
      const std::vector<float> pixels = runtime.preprocess(image.view());
      std::ofstream out(argv[4], std::ios::binary);
      out << s.image_width << ' ' << s.image_height << ' ' << (s.color ? 3 : 1) << " f32\n";
      out.write(reinterpret_cast<const char*>(pixels.data()),
                static_cast<std::streamsize>(pixels.size() * sizeof(float)));
      if (!out) throw std::runtime_error(std::string("cannot write ") + argv[4]);
    } else {
      return usage();
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "vo_preprocess: %s\n", error.what());
    return 1;
  }
  return 0;
}
