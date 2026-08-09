#include "ratingmanager.h"

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSaveFile>
#include <QXmlStreamReader>

RatingManager::RatingManager(QObject *parent) : QObject(parent) { }

QString RatingManager::sidecarPath(const QString &imagePath)
{
    const QFileInfo info(imagePath);
    // "photo.jpg" -> "photo.xmp", which is where Bridge and Photo Mechanic
    // look. (Lightroom writes sidecars only for raw files, but reads this.)
    return info.absolutePath() + "/" + info.completeBaseName() + ".xmp";
}

int RatingManager::rating(const QString &imagePath)
{
    if (imagePath.isEmpty())
        return Unrated;

    const auto cached = cache.constFind(imagePath);
    if (cached != cache.constEnd())
        return *cached;

    const int value = readSidecar(sidecarPath(imagePath));
    cache.insert(imagePath, value);
    return value;
}

bool RatingManager::setRating(const QString &imagePath, int rating)
{
    if (imagePath.isEmpty())
        return false;

    rating = qBound(Rejected, rating, MaxStars);
    if (this->rating(imagePath) == rating)
        return true;

    if (!writeSidecar(sidecarPath(imagePath), rating))
        return false;

    cache.insert(imagePath, rating);
    emit ratingChanged(imagePath, rating);
    return true;
}

void RatingManager::invalidate(const QString &imagePath)
{
    if (imagePath.isEmpty())
        cache.clear();
    else
        cache.remove(imagePath);
}

int RatingManager::readSidecar(const QString &sidecar)
{
    QFile file(sidecar);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return Unrated;

    // xmp:Rating appears either as an attribute on rdf:Description or as a
    // child element, depending on which tool wrote the file. Both are valid
    // and both turn up in the wild, so handle either.
    QXmlStreamReader xml(&file);
    while (!xml.atEnd()) {
        if (xml.readNext() != QXmlStreamReader::StartElement)
            continue;

        const auto attributes = xml.attributes();
        if (attributes.hasAttribute(QStringLiteral("xmp:Rating")))
            return attributes.value(QStringLiteral("xmp:Rating")).toInt();

        if (xml.name() == QStringLiteral("Rating")
            && xml.namespaceUri() == QStringLiteral("http://ns.adobe.com/xap/1.0/")) {
            return xml.readElementText().trimmed().toInt();
        }
    }
    return Unrated;
}

bool RatingManager::writeSidecar(const QString &sidecar, int rating)
{
    QFile existing(sidecar);
    QString content;
    if (existing.open(QIODevice::ReadOnly | QIODevice::Text)) {
        content = QString::fromUtf8(existing.readAll());
        existing.close();
    }

    if (content.isEmpty()) {
        // No sidecar yet: write a minimal, well-formed XMP packet.
        //
        // The xpacket header opens with U+FEFF. It is spliced in as a QChar
        // rather than written into the literal: as raw UTF-8 bytes in a
        // QString it gets encoded a second time on the way out and lands in
        // the file as "ï»¿", and as a source-file character it depends on the
        // compiler's source charset. QChar is unambiguous either way.
        content = QStringLiteral(
                          "<?xpacket begin=\"%1\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
                          "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"iqView\">\n"
                          " <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
                          "  <rdf:Description rdf:about=\"\"\n"
                          "    xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"\n"
                          "   xmp:Rating=\"%2\"/>\n"
                          " </rdf:RDF>\n"
                          "</x:xmpmeta>\n"
                          "<?xpacket end=\"w\"?>\n")
                          .arg(QChar(0xFEFF))
                          .arg(rating);
    } else {
        // A sidecar written by another tool can hold keywords, captions, edit
        // history and much else. Patch the rating in place rather than
        // regenerating the file, so nothing the user cares about is discarded.
        static const QRegularExpression attributeForm(
                QStringLiteral("xmp:Rating\\s*=\\s*\"[^\"]*\""));
        static const QRegularExpression elementForm(
                QStringLiteral("<xmp:Rating>[^<]*</xmp:Rating>"));

        if (content.contains(attributeForm)) {
            content.replace(attributeForm, QStringLiteral("xmp:Rating=\"%1\"").arg(rating));
        } else if (content.contains(elementForm)) {
            content.replace(elementForm,
                            QStringLiteral("<xmp:Rating>%1</xmp:Rating>").arg(rating));
        } else {
            // Present but with no rating recorded yet: inject the attribute
            // into the rdf:Description open tag, declaring the xmp namespace
            // alongside it if the file doesn't already bind it.
            static const QRegularExpression descriptionOpen(
                    QStringLiteral("<rdf:Description\\b"));
            const QRegularExpressionMatch match = descriptionOpen.match(content);
            if (!match.hasMatch())
                return false;   // not a shape we understand; refuse rather than corrupt

            QString injection = QStringLiteral("<rdf:Description");
            if (!content.contains(QStringLiteral("xmlns:xmp=")))
                injection += QStringLiteral(" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"");
            injection += QStringLiteral(" xmp:Rating=\"%1\"").arg(rating);
            content.replace(match.capturedStart(), match.capturedLength(), injection);
        }
    }

    // QSaveFile so an interrupted write can't leave a truncated sidecar behind
    // and lose metadata that was already there.
    QSaveFile out(sidecar);
    if (!out.open(QIODevice::WriteOnly | QIODevice::Text))
        return false;
    out.write(content.toUtf8());
    return out.commit();
}
