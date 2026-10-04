// Image conversion and resizing, bit-identical to the Python path the model
// was trained with (Pillow's convert() and resize(BILINEAR); OpenCV remap for
// undistortion lives in vo_stream.cpp next to the maps it uses).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vo_stream.hpp"

namespace vo {

ImageView ImageU8::view() const {
  ImageView v;
  v.data = data.data();
  v.width = width;
  v.height = height;
  v.channels = channels;
  v.stride = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  v.bgr = false;
  return v;
}

namespace {

void checkView(const ImageView& image) {
  if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
    throw std::invalid_argument("image: empty frame");
  }
  if (image.channels != 1 && image.channels != 3 && image.channels != 4) {
    throw std::invalid_argument("image: channels must be 1 (gray), 3 (colour) or 4 (colour + alpha)");
  }
  const std::size_t row = static_cast<std::size_t>(image.width) * image.channels;
  if (image.stride != 0 && image.stride < row) {
    throw std::invalid_argument("image: stride smaller than one row");
  }
}

}  // namespace

ImageU8 convertMode(const ImageView& image, bool color) {
  checkView(image);
  const std::size_t stride =
      image.stride != 0 ? image.stride : static_cast<std::size_t>(image.width) * image.channels;
  ImageU8 out;
  out.width = image.width;
  out.height = image.height;
  out.channels = color ? 3 : 1;
  out.data.resize(static_cast<std::size_t>(out.width) * out.height * out.channels);
  const int r_index = image.bgr ? 2 : 0;
  const int b_index = image.bgr ? 0 : 2;
  const int step = image.channels;
  for (int y = 0; y < image.height; ++y) {
    const std::uint8_t* src = image.data + static_cast<std::size_t>(y) * stride;
    std::uint8_t* dst = out.data.data() + static_cast<std::size_t>(y) * out.width * out.channels;
    for (int x = 0; x < image.width; ++x) {
      if (step == 1) {
        const std::uint8_t g = src[x];
        if (color) {
          // Pillow L -> RGB (l2rgb) replicates the value into all three bands.
          dst[3 * x + 0] = g;
          dst[3 * x + 1] = g;
          dst[3 * x + 2] = g;
        } else {
          dst[x] = g;
        }
      } else {
        // RGB or RGBA (alpha dropped: Pillow's rgba2rgb, and rgb2l for RGBA -> L).
        const std::uint8_t r = src[step * x + r_index];
        const std::uint8_t g = src[step * x + 1];
        const std::uint8_t b = src[step * x + b_index];
        if (color) {
          dst[3 * x + 0] = r;
          dst[3 * x + 1] = g;
          dst[3 * x + 2] = b;
        } else {
          // Pillow RGB -> L (Convert.c, rgb2l): ITU-R 601-2 luma in 16-bit
          // fixed point, rounded. 19595 + 38470 + 7471 = 65536.
          const std::uint32_t l = static_cast<std::uint32_t>(r) * 19595u +
                                  static_cast<std::uint32_t>(g) * 38470u +
                                  static_cast<std::uint32_t>(b) * 7471u + 0x8000u;
          dst[x] = static_cast<std::uint8_t>(l >> 16);
        }
      }
    }
  }
  return out;
}

namespace {

// ---- Pillow Resample.c, bilinear filter, 8 bits per channel ----------------

constexpr int kPrecisionBits = 32 - 8 - 2;

double bilinearFilter(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}
constexpr double kBilinearSupport = 1.0;

// precompute_coeffs(): per output index, the first input index (bounds[2i]),
// the number of inputs (bounds[2i+1]) and ksize normalised weights.
int precomputeCoeffs(int in_size, float in0, float in1, int out_size, std::vector<int>& bounds,
                     std::vector<double>& kk) {
  double filterscale;
  double scale;
  filterscale = scale = static_cast<double>(in1 - in0) / out_size;
  if (filterscale < 1.0) filterscale = 1.0;
  const double support = kBilinearSupport * filterscale;
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  kk.assign(static_cast<std::size_t>(out_size) * ksize, 0.0);
  bounds.assign(static_cast<std::size_t>(out_size) * 2, 0);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);  // C truncation, as Pillow
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &kk[static_cast<std::size_t>(xx) * ksize];
    int x = 0;
    for (; x < xmax; ++x) {
      const double w = bilinearFilter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; ++x) {
      if (ww != 0.0) k[x] /= ww;
    }
    for (; x < ksize; ++x) k[x] = 0;
    bounds[static_cast<std::size_t>(xx) * 2 + 0] = xmin;
    bounds[static_cast<std::size_t>(xx) * 2 + 1] = xmax;
  }
  return ksize;
}

// normalize_coeffs_8bpc(): double weights -> fixed point, rounded half away
// from zero by truncating +-0.5, exactly as Pillow does.
std::vector<std::int32_t> normalizeCoeffs8bpc(const std::vector<double>& prekk) {
  std::vector<std::int32_t> kk(prekk.size());
  for (std::size_t x = 0; x < prekk.size(); ++x) {
    if (prekk[x] < 0) {
      kk[x] = static_cast<std::int32_t>(-0.5 + prekk[x] * (1 << kPrecisionBits));
    } else {
      kk[x] = static_cast<std::int32_t>(0.5 + prekk[x] * (1 << kPrecisionBits));
    }
  }
  return kk;
}

inline std::uint8_t clip8(std::int32_t in) {
  if (in >= (1 << kPrecisionBits << 8)) return 255;
  if (in <= 0) return 0;
  return static_cast<std::uint8_t>(in >> kPrecisionBits);
}

// ImagingResampleHorizontal_8bpc: rows [offset, offset + out.height) of `in`.
void resampleHorizontal(ImageU8& out, const ImageU8& in, int offset, int ksize,
                        const std::vector<int>& bounds, const std::vector<double>& prekk) {
  const std::vector<std::int32_t> kk = normalizeCoeffs8bpc(prekk);
  const int channels = in.channels;
  for (int yy = 0; yy < out.height; ++yy) {
    const std::uint8_t* src = in.data.data() + static_cast<std::size_t>(yy + offset) * in.width * channels;
    std::uint8_t* dst = out.data.data() + static_cast<std::size_t>(yy) * out.width * channels;
    for (int xx = 0; xx < out.width; ++xx) {
      const int xmin = bounds[static_cast<std::size_t>(xx) * 2 + 0];
      const int xmax = bounds[static_cast<std::size_t>(xx) * 2 + 1];
      const std::int32_t* k = &kk[static_cast<std::size_t>(xx) * ksize];
      for (int c = 0; c < channels; ++c) {
        std::int32_t ss = 1 << (kPrecisionBits - 1);
        for (int x = 0; x < xmax; ++x) {
          ss += static_cast<std::int32_t>(src[(x + xmin) * channels + c]) * k[x];
        }
        dst[xx * channels + c] = clip8(ss);
      }
    }
  }
}

// ImagingResampleVertical_8bpc.
void resampleVertical(ImageU8& out, const ImageU8& in, int ksize, const std::vector<int>& bounds,
                      const std::vector<double>& prekk) {
  const std::vector<std::int32_t> kk = normalizeCoeffs8bpc(prekk);
  const int channels = in.channels;
  const std::size_t row = static_cast<std::size_t>(in.width) * channels;
  for (int yy = 0; yy < out.height; ++yy) {
    const std::int32_t* k = &kk[static_cast<std::size_t>(yy) * ksize];
    const int ymin = bounds[static_cast<std::size_t>(yy) * 2 + 0];
    const int ymax = bounds[static_cast<std::size_t>(yy) * 2 + 1];
    std::uint8_t* dst = out.data.data() + static_cast<std::size_t>(yy) * row;
    for (std::size_t xc = 0; xc < row; ++xc) {
      std::int32_t ss = 1 << (kPrecisionBits - 1);
      for (int y = 0; y < ymax; ++y) {
        ss += static_cast<std::int32_t>(in.data[static_cast<std::size_t>(y + ymin) * row + xc]) * k[y];
      }
      dst[xc] = clip8(ss);
    }
  }
}

}  // namespace

ImageU8 resizeBilinearPIL(const ImageU8& image, int width, int height) {
  if (image.channels != 1 && image.channels != 3) {
    throw std::invalid_argument("resize: channels must be 1 or 3");
  }
  if (width <= 0 || height <= 0) throw std::invalid_argument("resize: size must be positive");
  // Image.resize returns a copy when the size already matches.
  if (width == image.width && height == image.height) return image;

  // ImagingResampleInner with box = (0, 0, xsize, ysize).
  const float box[4] = {0.0f, 0.0f, static_cast<float>(image.width), static_cast<float>(image.height)};
  const bool need_horizontal = width != image.width || box[0] != 0.0f || box[2] != width;
  const bool need_vertical = height != image.height || box[1] != 0.0f || box[3] != height;

  std::vector<int> bounds_horiz;
  std::vector<int> bounds_vert;
  std::vector<double> kk_horiz;
  std::vector<double> kk_vert;
  const int ksize_horiz = precomputeCoeffs(image.width, box[0], box[2], width, bounds_horiz, kk_horiz);
  const int ksize_vert = precomputeCoeffs(image.height, box[1], box[3], height, bounds_vert, kk_vert);

  // First and last source rows the vertical pass will read.
  const int ybox_first = bounds_vert[0];
  const int ybox_last = bounds_vert[static_cast<std::size_t>(height) * 2 - 2] +
                        bounds_vert[static_cast<std::size_t>(height) * 2 - 1];

  ImageU8 temp;
  const ImageU8* vertical_input = &image;  // the original, or the horizontally resampled rows
  if (need_horizontal) {
    // Shift the vertical bounds to the rows the horizontal pass produces.
    for (int i = 0; i < height; ++i) bounds_vert[static_cast<std::size_t>(i) * 2] -= ybox_first;
    temp.width = width;
    temp.height = ybox_last - ybox_first;
    temp.channels = image.channels;
    temp.data.assign(static_cast<std::size_t>(temp.width) * temp.height * temp.channels, 0);
    resampleHorizontal(temp, image, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
    vertical_input = &temp;
  }
  if (need_vertical) {
    ImageU8 out;
    out.width = vertical_input->width;
    out.height = height;
    out.channels = vertical_input->channels;
    out.data.assign(static_cast<std::size_t>(out.width) * out.height * out.channels, 0);
    resampleVertical(out, *vertical_input, ksize_vert, bounds_vert, kk_vert);
    return out;
  }
  return temp;  // need_horizontal only (equal sizes returned above)
}

ImageU8 loadImageFile(const std::string& path, bool color) {
  // IMREAD_UNCHANGED: the file's own channels, and no EXIF rotation - Pillow's
  // open() does not apply the orientation tag either (IMREAD_COLOR would).
  // OpenCV hands colour over as B, G, R(, A); convertMode then gives Pillow's
  // RGB or L. JPEGs decode to the same pixels Pillow gets (both libjpeg-turbo,
  // the default integer IDCT and fancy upsampling).
  const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (image.empty()) throw std::runtime_error("cannot read image " + path);
  if (image.depth() != CV_8U) throw std::runtime_error("not an 8-bit image: " + path);
  if (image.channels() != 1 && image.channels() != 3 && image.channels() != 4) {
    throw std::runtime_error("unsupported channel count in " + path);
  }
  ImageView view;
  view.data = image.data;
  view.width = image.cols;
  view.height = image.rows;
  view.channels = image.channels();
  view.stride = image.step[0];
  view.bgr = true;
  return convertMode(view, color);
}

}  // namespace vo
