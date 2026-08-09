#include "ratingmanager.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QtTest>

// The sidecar writer edits files other applications own, so the cases that
// matter most here are the ones where getting it wrong would silently destroy
// a user's keywords or captions.
class RatingManagerTests : public QObject
{
    Q_OBJECT

private slots:
    void sidecarPathReplacesExtension();
    void roundTripsRating();
    void roundTripsRejection();
    void readsAttributeForm();
    void readsElementForm();
    void preservesForeignMetadata();
    void injectsIntoSidecarWithoutRating();
    void missingSidecarIsUnrated();

private:
    static QString write(const QDir &dir, const QString &name, const QString &content)
    {
        const QString path = dir.filePath(name);
        QFile file(path);
        file.open(QIODevice::WriteOnly | QIODevice::Text);
        file.write(content.toUtf8());
        file.close();
        return path;
    }

    static QString read(const QString &path)
    {
        QFile file(path);
        file.open(QIODevice::ReadOnly | QIODevice::Text);
        return QString::fromUtf8(file.readAll());
    }
};

void RatingManagerTests::sidecarPathReplacesExtension()
{
    const QString sidecar = RatingManager::sidecarPath("/photos/DSC_0001.jpg");
    QCOMPARE(QFileInfo(sidecar).fileName(), QStringLiteral("DSC_0001.xmp"));
}

void RatingManagerTests::roundTripsRating()
{
    QTemporaryDir dir;
    const QString image = dir.filePath("a.jpg");

    RatingManager manager;
    QCOMPARE(manager.rating(image), RatingManager::Unrated);
    QVERIFY(manager.setRating(image, 4));
    QCOMPARE(manager.rating(image), 4);

    // A second manager must see it too -- otherwise it only lived in the cache.
    RatingManager fresh;
    QCOMPARE(fresh.rating(image), 4);
}

void RatingManagerTests::roundTripsRejection()
{
    QTemporaryDir dir;
    const QString image = dir.filePath("b.jpg");

    RatingManager manager;
    QVERIFY(manager.setRating(image, RatingManager::Rejected));

    RatingManager fresh;
    QCOMPARE(fresh.rating(image), RatingManager::Rejected);
    // -1 is the Bridge/Lightroom convention for a rejection.
    QVERIFY(read(RatingManager::sidecarPath(image)).contains(QStringLiteral("\"-1\"")));
}

void RatingManagerTests::readsAttributeForm()
{
    QTemporaryDir dir;
    QDir d(dir.path());
    write(d, "c.xmp",
          "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">"
          "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
          "<rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\""
          " xmp:Rating=\"3\"/></rdf:RDF></x:xmpmeta>");

    RatingManager manager;
    QCOMPARE(manager.rating(d.filePath("c.jpg")), 3);
}

void RatingManagerTests::readsElementForm()
{
    QTemporaryDir dir;
    QDir d(dir.path());
    write(d, "e.xmp",
          "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">"
          "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
          "<rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\">"
          "<xmp:Rating>2</xmp:Rating>"
          "</rdf:Description></rdf:RDF></x:xmpmeta>");

    RatingManager manager;
    QCOMPARE(manager.rating(d.filePath("e.jpg")), 2);
}

// The important one: rewriting a rating must not throw away anything else the
// sidecar carries.
void RatingManagerTests::preservesForeignMetadata()
{
    QTemporaryDir dir;
    QDir d(dir.path());
    write(d, "f.xmp",
          "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">"
          "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
          "<rdf:Description rdf:about=\"\""
          " xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\""
          " xmlns:dc=\"http://purl.org/dc/elements/1.1/\""
          " xmp:Rating=\"1\">"
          "<dc:subject><rdf:Bag><rdf:li>sunset</rdf:li><rdf:li>iceland</rdf:li></rdf:Bag>"
          "</dc:subject>"
          "<dc:description><rdf:Alt><rdf:li xml:lang=\"x-default\">A caption</rdf:li>"
          "</rdf:Alt></dc:description>"
          "</rdf:Description></rdf:RDF></x:xmpmeta>");

    RatingManager manager;
    QVERIFY(manager.setRating(d.filePath("f.jpg"), 5));

    const QString after = read(d.filePath("f.xmp"));
    QVERIFY2(after.contains(QStringLiteral("sunset")), "keyword was discarded");
    QVERIFY2(after.contains(QStringLiteral("iceland")), "keyword was discarded");
    QVERIFY2(after.contains(QStringLiteral("A caption")), "caption was discarded");
    QVERIFY(after.contains(QStringLiteral("xmp:Rating=\"5\"")));
    QVERIFY2(!after.contains(QStringLiteral("xmp:Rating=\"1\"")), "old rating still present");
}

void RatingManagerTests::injectsIntoSidecarWithoutRating()
{
    QTemporaryDir dir;
    QDir d(dir.path());
    write(d, "g.xmp",
          "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">"
          "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
          "<rdf:Description rdf:about=\"\""
          " xmlns:dc=\"http://purl.org/dc/elements/1.1/\">"
          "<dc:creator><rdf:Seq><rdf:li>Someone</rdf:li></rdf:Seq></dc:creator>"
          "</rdf:Description></rdf:RDF></x:xmpmeta>");

    RatingManager manager;
    QVERIFY(manager.setRating(d.filePath("g.jpg"), 2));

    const QString after = read(d.filePath("g.xmp"));
    QVERIFY2(after.contains(QStringLiteral("Someone")), "creator was discarded");
    QVERIFY(after.contains(QStringLiteral("xmp:Rating=\"2\"")));
    // The namespace has to be declared or the file is not valid XMP.
    QVERIFY(after.contains(QStringLiteral("xmlns:xmp=")));

    RatingManager fresh;
    QCOMPARE(fresh.rating(d.filePath("g.jpg")), 2);
}

void RatingManagerTests::missingSidecarIsUnrated()
{
    QTemporaryDir dir;
    RatingManager manager;
    QCOMPARE(manager.rating(QDir(dir.path()).filePath("nope.jpg")), RatingManager::Unrated);
}

QTEST_MAIN(RatingManagerTests)
#include "tst_ratingmanagertests.moc"
