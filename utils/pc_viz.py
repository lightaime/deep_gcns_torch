import vtk
import numpy as np
import random
import os

print('Using', vtk.vtkVersion.GetVTKSourceVersion())


class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent, pointcloud):
        self.parent = parent
        self.pointcloud = pointcloud
        self.AddObserver("KeyPressEvent", self.keyPressEvent)

    def keyPressEvent(self, obj, event):
        key = self.parent.GetKeySym()
        if key == '+':
            point_size = self.pointcloud.vtkActor.GetProperty().GetPointSize()
            self.pointcloud.vtkActor.GetProperty().SetPointSize(point_size + 1)
            print(str(point_size) + " " + key)
        return


class VtkPointCloud:

    def __init__(self, point_size=18, maxNumPoints=1e8):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clear_points()

        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("Colors")

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)

        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetPointSize(point_size)

    def add_point(self, point, color):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.colors.InsertNextTuple(color)
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            print("VIZ: Reached max number of points!")
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkPolyData.GetPointData().SetScalars(self.colors)
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clear_points(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


def getActorCircle(radius_inner=100, radius_outer=99, color=(1, 0, 0)):
    """"""
    # create source
    source = vtk.vtkDiskSource()
    source.SetInnerRadius(radius_inner)
    source.SetOuterRadius(radius_outer)
    source.SetRadialResolution(100)
    source.SetCircumferentialResolution(100)

    # Transformer
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(90, 1, 0, 0)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputConnection(source.GetOutputPort())
    transformFilter.Update()

    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transformFilter.GetOutputPort())

    # actor
    actor = vtk.vtkActor()
    actor.GetProperty().SetColor(color)
    actor.SetMapper(mapper)

    return actor


def show_pointclouds(points, colors, text=[], title="Default", png_path="", interactive=True, orientation='horizontal'):
    """
    Show multiple point clouds specified as lists. First clouds at the bottom.
    :param points: list of pointclouds, item: numpy (N x 3) XYZ
    :param colors: list of corresponding colors, item: numpy (N x 3) RGB [0..255]
    :param title: window title
    :param text: text per point cloud
    :param png_path: where to save png image
    :param interactive: wether to display window or not, useful if you only want to take screenshot
    :return: nothing
    """

    # make sure pointclouds is a list
    assert isinstance(points, type([])), \
        "Pointclouds argument must be a list"

    # make sure colors is a list
    assert isinstance(colors, type([])), \
        "Colors argument must be a list"

    # make sure number of pointclouds and colors are the same
    assert len(points) == len(colors), \
        "Number of pointclouds (%d) is different then number of colors (%d)" % (len(points), len(colors))

    while len(text) < len(points):
        text.append("")

    # Number of pointclouds to be displayed in this window
    num_pointclouds = len(points)

    point_size = 10
    pointclouds = [VtkPointCloud(point_size) for _ in range(num_pointclouds)]
    renderers = [vtk.vtkRenderer() for _ in range(num_pointclouds)]

    height = 1.0 / max(num_pointclouds, 1)
    viewports = [(i*height, (i+1)*height) for i in range(num_pointclouds)]
    #print(viewports)

    # iterate over all point clouds
    for i, pc in enumerate(points):
        pc = pc.squeeze()
        co = colors[i].squeeze()
        assert pc.shape[0] == co.shape[0], \
            "expected same number of points (%d) then colors (%d), cloud index = %d" % (pc.shape[0], co.shape[0], i)
        assert pc.shape[1] == 3, "expected points to be N x 3, got N x %d" % pc.shape[1]
        assert co.shape[1] == 3, "expected colors to be N x 3, got N x %d" % co.shape[1]

        # for each point cloud iterate over all points
        for j in range(pc.shape[0]):
            point = pc[j, :]
            color = co[j, :]
            pointclouds[i].add_point(point, color)

        renderers[i].AddActor(pointclouds[i].vtkActor)
        # renderers[i].AddActor(vtk.vtkAxesActor())
        renderers[i].SetBackground(1.0, 1.0, 1.0)
        if orientation == 'horizontal':
            print(viewports[i][0])
            renderers[i].SetViewport(viewports[i][0], 0.0, viewports[i][1], 1.0)
        elif orientation == 'vertical':
            renderers[i].SetViewport(0.0, viewports[i][0], 1.0, viewports[i][1])
        else:
            raise Exception('Not a valid orientation!')
        renderers[i].ResetCamera()

    # Add circle to first render
    renderers[0].AddActor(getActorCircle())
    renderers[0].AddActor(getActorCircle(50, 49, color=(0, 1, 0)))

    # Text actors
    text_actors = [vtk.vtkTextActor() for _ in text]
    for i, ta in enumerate(text_actors):
        if orientation == 'horizontal':
            ta.SetInput('                ' + text[i])
        elif orientation == 'vertical':
            ta.SetInput(text[i] + '\n\n\n\n\n\n')
        else:
            raise Exception('Not a valid orientation!')
        txtprop = ta.GetTextProperty()
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(0)
        txtprop.SetColor(0, 0, 0)
        # txtprop.SetJustificationToCentered()
        # ta.SetDisplayPosition(500, 10)
        # ta.SetAlignmentPoint()
        renderers[i].AddActor(ta)

    # Render Window
    render_window = vtk.vtkRenderWindow()
    for renderer in renderers:
        render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    render_window_interactor.SetRenderWindow(render_window)

    [center_x, center_y, center_z] = np.mean(points[0].squeeze(), axis=0)
    camera = vtk.vtkCamera()
    # d = 10
    # camera.SetViewUp(0, -1, 0)

    # camera.SetPosition(center_x + d, center_y + d, center_z + d / 2)
    # camera.SetFocalPoint(center_x, center_y, center_z)
    # camera.SetFocalPoint(0, 0, 0)

    camera.SetViewUp(0, 0, 1)
    if orientation == 'horizontal':
        camera.SetPosition(3, -10, 2)
        camera.SetFocalPoint(3, 1.5, 1.5)
    elif orientation == 'vertical':
        camera.SetPosition(1.5, -6, 2)
        camera.SetFocalPoint(1.5, 1.5, 1.5)
    else:
        raise Exception('Not a valid orientation!')

    camera.SetClippingRange(0.002, 1000)
    for renderer in renderers:
        renderer.SetActiveCamera(camera)

    # Begin Interaction
    render_window.Render()
    render_window.SetWindowName(title)
    if orientation == 'horizontal':
        render_window.SetSize(1940, 720)
    elif orientation == 'vertical':
        render_window.SetSize(600, 1388)
    else:
        raise Exception('Not a valid orientation!')

    if interactive:
        render_window_interactor.Start()

    if png_path:
        # screenshot code:
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(render_window)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(png_path)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()


def get_points_colors_from_obj(filename, limit=1):
    points = []
    colors = []
    with open(filename) as f:
        for line in f:
            parts = line.strip().split()
            points.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
            colors.append(np.array([float(parts[4]), float(parts[5]), float(parts[6])]))
    points = np.array(points)
    colors = np.array(colors)
    idx = points[:, 1] >= limit
    return points[idx, :], colors[idx, :]


def visualize_part_seg(file_name_pred, file_name_gt, comparison_folder_list, limit=1, text=[], png_path="",
                          interactive=True, orientation='horizontal'):
    # load base point cloud
    gt_points, gt_colors = get_points_colors_from_obj(os.path.join(comparison_folder_list[0], file_name_gt), limit)

    idx_gt = gt_points[:, 1] >= limit

    all_points = [gt_points[idx_gt, :3]]
    all_colors = [gt_colors[idx_gt, :3]]

    for folder in comparison_folder_list:
        pts, col = get_points_colors_from_obj(os.path.join(folder, file_name_pred), limit=limit)

        all_points.append(pts)
        all_colors.append(col)

    print(np.asarray(all_points).shape)
    show_pointclouds(all_points, all_colors, text=text, png_path=png_path, interactive=interactive,
                     orientation=orientation)

