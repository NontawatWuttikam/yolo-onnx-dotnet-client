// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Numerics.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Drawing;
using SixLabors.Fonts;

const string imagePath = "/Users/nontawatwuttikam/ConsoleApp1/ConsoleApp1/data/catdogperson.jpg";
const string modelPath = "/Users/nontawatwuttikam/ConsoleApp1/ConsoleApp1/onnx_models/yolov8n.onnx";
// input will be fit into inputSize x inputSize with padding
const int inputSize = 1280;
const float nmsThreshold = 0.45f;
const float confidenceThreshold = 0.4f;
const bool saveImage = true;

// ----------------- Image Preprocessing -----------------
using Image<Rgb24> image = Image.Load<Rgb24>(imagePath);

int originalWidth = image.Width;
int originalHeight = image.Height;

float newScale = Math.Min((float)inputSize / image.Width, (float)inputSize / image.Height);

Image<Rgb24> originalImage = image.Clone();

image.Mutate(x => x.Resize((int)(image.Width * newScale), (int)(image.Height * newScale)));

image.Mutate(x =>
{
    x.Resize(new ResizeOptions
    {
        Size = new Size(inputSize, inputSize),
        Mode = ResizeMode.Pad,
        PadColor = Color.Gray,
        Position = AnchorPositionMode.TopLeft
    });
});

DenseTensor<float> processedImage = new(new[] { 1, 3, inputSize, inputSize });

image.ProcessPixelRows(accessor =>
{
    for (int y = 0; y < accessor.Height; y++)
    {
        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
        for (int x = 0; x < accessor.Width; x++)
        {
            processedImage[0, 0, y, x] = pixelSpan[x].R / 255f;
            processedImage[0, 1, y, x] = pixelSpan[x].G / 255f;
            processedImage[0, 2, y, x] = pixelSpan[x].B / 255f;
        }
    }
});

// ----------------- Run Inference -----------------
using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
    OrtMemoryInfo.DefaultInstance,
    processedImage.Buffer,
    new long[] { 1, 3, inputSize, inputSize });

var inputs = new Dictionary<string, OrtValue>
{
    { "images", inputOrtValue }
};

using var session = new InferenceSession(modelPath);
using var runOptions = new RunOptions();
using IDisposableReadOnlyCollection<OrtValue> results = session.Run(runOptions, inputs, session.OutputNames);

OrtValue output = results.First().Value;
var shape = output.GetTensorTypeAndShape().Shape;
var span = output.GetTensorDataAsSpan<float>();

int batch = (int)shape[0];
int channels = (int)shape[1];
int numBoxes = (int)shape[2];

// ----------------- Parse Detections -----------------
List<Detection> detections = new();

for (int i = 0; i < numBoxes; i++)
{
    float x = span[0 * numBoxes + i];
    float y = span[1 * numBoxes + i];
    float w = span[2 * numBoxes + i];
    float h = span[3 * numBoxes + i];

    int bestClass = -1;
    float bestScore = 0f;
    for (int c = 4; c < channels; c++)
    {
        float score = span[c * numBoxes + i];
        if (score > bestScore)
        {
            bestScore = score;
            bestClass = c - 4;
        }
    }

    if (bestScore > confidenceThreshold) // confidence threshold
    {
        float x1 = x - w / 2;
        float y1 = y - h / 2;
        float x2 = x + w / 2;
        float y2 = y + h / 2;

        detections.Add(new Detection
        {
            X1 = x1,
            Y1 = y1,
            X2 = x2,
            Y2 = y2,
            Score = bestScore,
            ClassId = bestClass
        });
    }
}

// ----------------- Apply NMS -----------------
var finalDetections = NmsHelper.NonMaxSuppression(detections, nmsThreshold  );

foreach (var d in finalDetections)
{
    d.X1 = d.X1 * inputSize / newScale;
    d.Y1 = d.Y1 * inputSize / newScale;
    d.X2 = d.X2 * inputSize / newScale;
    d.Y2 = d.Y2 * inputSize / newScale;

    Console.WriteLine($"Class={d.ClassId}, Score={d.Score}, " +
                      $"BBox=({d.X1},{d.Y1},{d.X2},{d.Y2})" +
                      $"Centroid=({(d.X1 + d.X2) / 2},{(d.Y1 + d.Y2) / 2})");
}

if (saveImage)
{
    // draw and save image
    originalImage.Mutate(ctx =>
    {
        foreach (var d in finalDetections)
        {
            var rect = new RectangleF(d.X1, d.Y1, d.X2 - d.X1, d.Y2 - d.Y1);
            ctx.Draw(Color.Red, 2, rect);

            var text = $"Class {d.ClassId} ({d.Score:P2})";
            var font = SystemFonts.CreateFont("Arial", 16);
            var textSize = TextMeasurer.MeasureSize(text, new TextOptions(font));
            var textPosition = new PointF(d.X1, d.Y1 - textSize.Height
    );
            ctx.DrawText(text, font, Color.White, textPosition);
        }
    });

    originalImage.Save("output.png");
    image.Save("processed.png");
}

public class Detection
{
    public float X1, Y1, X2, Y2;
    public float Score;
    public int ClassId;
}

public static class NmsHelper
{
    public static List<Detection> NonMaxSuppression(
        List<Detection> detections,
        float iouThreshold = 0.45f)
    {
        var results = new List<Detection>();

        foreach (var group in detections.GroupBy(d => d.ClassId))
        {
            var dets = group.OrderByDescending(d => d.Score).ToList();
            while (dets.Count > 0)
            {
                var best = dets[0];
                results.Add(best);
                dets.RemoveAt(0);

                dets = dets.Where(d => IoU(best, d) < iouThreshold).ToList();
            }
        }

        return results;
    }

    private static float IoU(Detection a, Detection b)
    {
        float interArea = Intersection(a, b);
        float unionArea = Area(a) + Area(b) - interArea;
        return unionArea == 0 ? 0 : interArea / unionArea;
    }

    private static float Area(Detection d)
        => Math.Max(0, d.X2 - d.X1) * Math.Max(0, d.Y2 - d.Y1);

    private static float Intersection(Detection a, Detection b)
    {
        float x1 = Math.Max(a.X1, b.X1);
        float y1 = Math.Max(a.Y1, b.Y1);
        float x2 = Math.Min(a.X2, b.X2);
        float y2 = Math.Min(a.Y2, b.Y2);
        return Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
    }
}
