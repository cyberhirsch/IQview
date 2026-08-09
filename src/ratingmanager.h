#ifndef RATINGMANAGER_H
#define RATINGMANAGER_H

#include <QHash>
#include <QObject>
#include <QString>

// Star ratings and rejections for the culling workflow, persisted as XMP
// sidecar files next to each image.
//
// XMP rather than a private database so ratings survive the round trip through
// Lightroom, Bridge, digiKam and Photo Mechanic -- culling in iqView and then
// finishing elsewhere is the whole point, and a rating that only exists inside
// this app would be worthless for that. The encoding follows the Adobe
// convention that every one of those tools already understands:
//
//   xmp:Rating =  1..5   stars
//   xmp:Rating =  0      unrated
//   xmp:Rating = -1      rejected
//
// Sidecars are written as "<image>.xmp" (extension replaced), matching what
// Bridge and Photo Mechanic look for.
class RatingManager : public QObject
{
    Q_OBJECT

public:
    static constexpr int Rejected = -1;
    static constexpr int Unrated = 0;
    static constexpr int MaxStars = 5;

    explicit RatingManager(QObject *parent = nullptr);

    // Cached; reads the sidecar on first request for a given file.
    int rating(const QString &imagePath);

    // Writes through to the sidecar. Returns false if the file could not be
    // written (read-only media, permissions).
    bool setRating(const QString &imagePath, int rating);

    // Path of the sidecar iqView would use for this image.
    static QString sidecarPath(const QString &imagePath);

    // Drops cached values so externally-edited sidecars are picked up again.
    void invalidate(const QString &imagePath = QString());

signals:
    void ratingChanged(const QString &imagePath, int rating);

private:
    static int readSidecar(const QString &sidecar);
    static bool writeSidecar(const QString &sidecar, int rating);

    QHash<QString, int> cache;
};

#endif // RATINGMANAGER_H
