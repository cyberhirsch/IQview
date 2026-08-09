#ifndef QVGRAPHICSVIEW_H
#define QVGRAPHICSVIEW_H

#include "qvimagecore.h"
#include <QGraphicsView>
#include <QImageReader>
#include <QMimeData>
#include <QDir>
#include <QTimer>
#include <QFileInfo>
#include <QLabel>
#include <QPointer>
#include <QProcess>

class QVGraphicsView : public QGraphicsView
{
    Q_OBJECT

public:
    QVGraphicsView(QWidget *parent = nullptr);

    enum class ScaleMode { resetScale, zoom };
    Q_ENUM(ScaleMode)

    enum class GoToFileMode { constant, first, previous, next, last };
    Q_ENUM(GoToFileMode)

    QMimeData *getMimeData() const;
    void loadMimeData(const QMimeData *mimeData);
    void loadFile(const QString &fileName);

    void reloadFile();

    void zoomIn(const QPoint &pos = QPoint(-1, -1));

    void zoomOut(const QPoint &pos = QPoint(-1, -1));

    void zoom(qreal scaleFactor, const QPoint &pos = QPoint(-1, -1));

    void scaleExpensively();
    void makeUnscaled();

    void resetScale();
    void originalSize();

    void goToFile(const GoToFileMode &mode, int index = 0);

    void settingsUpdated();

    void closeImage();
    void jumpToNextFrame();
    enum class RetouchTool { Off, Brush, Lasso };
    void setPaused(const bool &desiredState);
    void setSpeed(const int &desiredSpeed);


    void rotateImage(int rotation);
    void toggleRetouchMode();
    void applyRetouch();
    bool undoRetouch();
    bool redoRetouch();
    void applyCreativeFill();
    void applyIsolate();
    void exitRetouchMode();
    bool checkGenerativeAccess();
    void changeBrushSize(int delta);
    void showAiLogWindow();
    void exportDebugReport();

    // Culling workflow: star ratings and rejections, stored as XMP sidecars.
    void setRatingForCurrentFile(int rating);
    int ratingForCurrentFile();
    void exportKeepers();

    const QVImageCore::FileDetails &getCurrentFileDetails() const
    {
        return imageCore.getCurrentFileDetails();
    }
    const QPixmap &getLoadedPixmap() const { return imageCore.getLoadedPixmap(); }

    // True while an AI result is displayed that has not been saved to a real
    // file yet (it lives only in the temp directory).
    bool hasUnsavedEdits() const { return !editedSource.isEmpty(); }
    // Absolute path of the original image the current edits came from.
    QString getEditedSourcePath() const { return editedSource; }
    void markEditsSaved() { editedSource.clear(); }
    const QMovie &getLoadedMovie() const { return imageCore.getLoadedMovie(); }

signals:
    void cancelSlideshow();

    void fileChanged();

    void updatedLoadedPixmapItem();
    void zoomChanged(qreal factor);

protected:
    void wheelEvent(QWheelEvent *event) override;

    void resizeEvent(QResizeEvent *event) override;

    void dropEvent(QDropEvent *event) override;

    void dragEnterEvent(QDragEnterEvent *event) override;

    void dragMoveEvent(QDragMoveEvent *event) override;

    void dragLeaveEvent(QDragLeaveEvent *event) override;

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    void enterEvent(QEvent *event) override;
#else
    void enterEvent(QEnterEvent *event) override;
#endif

    void mousePressEvent(QMouseEvent *event) override;

    void mouseMoveEvent(QMouseEvent *event) override;

    void mouseReleaseEvent(QMouseEvent *event) override;

    bool event(QEvent *event) override;

    void drawForeground(QPainter *painter, const QRectF &rect) override;

    void fitInViewMarginless(const QRectF &rect);
    void fitInViewMarginless(const QGraphicsItem *item);

    void centerOn(const QPointF &pos);

    void centerOn(qreal x, qreal y);

    void centerOn(const QGraphicsItem *item);

private slots:
    void animatedFrameChanged(QRect rect);

    void postLoad();

    void updateLoadedPixmapItem();

private:
    void updateFilteringMode();

    QGraphicsPixmapItem *loadedPixmapItem;

    constexpr static int MARGIN = -2;
    constexpr static qreal MAX_EXPENSIVE_SCALING_SIZE = 3;

    // Set to too high a value to activate for now...
    constexpr static qreal MAX_FILTERING_SIZE = 5000;

    qreal currentScale;
    QSize scaledSize;
    bool isOriginalSize;
    QPoint lastZoomEventPos;
    QPointF lastZoomRoundingError;
    QPointF lastScrollRoundingError;

    QTransform absoluteTransform;
    QTransform zoomBasis;
    qreal zoomBasisScaleFactor;

    QVImageCore imageCore{ this };
    class RatingManager *ratingManager = nullptr;

    // Brief on-screen confirmation after a rating key, so culling can be done
    // by feel without looking away from the image.
    QLabel *ratingLabel = nullptr;
    QTimer *ratingLabelTimer = nullptr;
    void showRatingFeedback(int rating);
    void repositionRatingLabel();

    QTimer *expensiveScaleTimerNew;
    QPointF centerPoint;
    Qt::MouseButton mousePressButton;
    Qt::KeyboardModifiers mousePressModifiers;
    QPoint mousePressPosition;

    // Retouching
    RetouchTool retouchTool = RetouchTool::Off;
    bool isDrawing = false;
    QImage maskImage;
    QGraphicsPixmapItem *maskItem = nullptr;
    int brushSize = 50;
    QPointF lastMouseScenePos;
    QPolygonF lassoPolygon;

    // Undo/redo history for AI edits. A single stored pixmap meant chaining
    // Retouch → Fill → Isolate threw away every state but the most recent, so
    // only the last step could ever be walked back. Depth is capped because
    // these are full-resolution pixmaps and each one costs real memory.
    static constexpr int MAX_UNDO_STEPS = 5;
    QList<QPixmap> undoStack;
    QList<QPixmap> redoStack;
    void pushUndoState(const QPixmap &pixmap);
    QPointer<QDialog> aiLogDialog;
    // Set by the AI handlers immediately before they load their temp output;
    // consumed by loadFile() into editedSource. Any load that doesn't set it
    // is a normal file open, which clears the edited state.
    QString pendingEditedSource;
    QString editedSource;
    void beginAiResultLoad(const QString &outputPath);

    // Batch Creative Fill: results accumulate here until the worker reports
    // BATCH_DONE, then the user picks one.
    int fluxBatchExpected = 1;
    QStringList fluxBatchResults;
    void finishFluxBatch();
    void updateMaskItem();
    void paintOnMask(const QPointF &scenePos);
    void finalizeLasso();
    bool isMaskEmpty() const;

    // Persistent AI Worker
    QProcess *workerProcess = nullptr;
    bool isWorkerReady = false;
    void ensureWorkerStarted();
    void handleWorkerOutput();
    QString pendingOutputPath;
    bool maskHasPaint = false;

    // Idle prefetch: quietly warm the LaMa worker a few seconds after the
    // user settles on an image, so pressing R has no cold-start delay.
    // Only fires if the AI environment is already set up (never triggers
    // first-run setup) and stays silent unless the user is actually waiting
    // on it (see silentWorkerStart).
    QTimer *idlePrefetchTimer = nullptr;
    bool silentWorkerStart = false;
    void scheduleIdlePrefetch();
    void performIdlePrefetch();

    void repositionPromptBar();
    void showAiStatus(const QString &text);
    void hideAiStatus();
    void repositionAiStatus();

    // Which AI worker currently owns a job, so Escape / the HUD's Cancel
    // button know what to interrupt. Long jobs (a 50-step Flux generation, a
    // multi-GB model download) were previously only escapable by killing the
    // whole application.
    enum class AiJob { None, Retouch, Fill, Isolate };
    AiJob activeAiJob = AiJob::None;
    bool cancelAiOperation();
    static QString resolveScriptsDir();
    static QString resolveVenvDir();
    static QString resolvePythonExe();
    static QString resolveUvDir();
    static QString resolveUvExe();
    // First-run Python environment setup, shared by all AI features.
    bool ensureAiEnvironment();
    static QString resolveEnvStampPath();
    static QString currentEnvStamp();
    bool confirmDiskSpace(const QString &targetDir, qint64 requiredBytes);
    // Stamp the user declined to update this session, so a refusal is honoured
    // for the rest of the session instead of re-prompting on every AI action.
    QString declinedEnvStamp;
    bool ensureUvInstalled(class QProgressDialog &progress, const QString &logPath);
    static QString resolveLogPath();
    static QString resolveModelsDir();

    // Generative
    void ensureFluxStarted();
    void handleFluxOutput();

    QProcess *fluxProcess = nullptr;
    QString fluxLoadedModelId;
    class RetouchPromptBar *promptBar = nullptr;

    // Isolate — SAM 3 background removal (subject cutout)
    enum class IsolateState { Idle, WaitingForResult };
    void ensureIsolateStarted();
    void handleIsolateOutput();

    QProcess     *isolateProcess = nullptr;
    IsolateState  isolateState   = IsolateState::Idle;
    QString       isolateInputPath;

    // AI status HUD (floating panel shown during model load / download / inference):
    // container holding the status text, an indeterminate progress bar, and a
    // Cancel button.
    QWidget *aiStatusWidget = nullptr;
    QLabel *aiStatusLabel = nullptr;
    class QProgressBar *aiProgressBar = nullptr;
    class QPushButton *aiCancelButton = nullptr;
};
#endif // QVGRAPHICSVIEW_H
